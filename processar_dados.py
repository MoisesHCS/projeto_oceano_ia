"""
Sistema de deteccao de padroes de corrente, previsao de deriva de poluente
com incerteza (Monte Carlo) e planejamento de rota de interceptacao via
Monte Carlo Tree Search (MCTS).

Linhas de pesquisa do roteiro de IA I cobertas:
  1. Aprendizado Nao Supervisionado -> Clustering (DBSCAN) de regimes de corrente
     (survey base: Jain, 2005 - Data Clustering: 50 Years Beyond K-Means)
  2. Planejamento de Trajetorias / Busca -> Monte Carlo Tree Search para a
     rota de interceptacao da embarcacao de contencao
     (survey base: Browne et al., 2012 - A Survey of Monte Carlo Tree Search
      Methods; Mandziuk, 2022 - MCTS: a review of recent modifications)
"""

import math
import random

import folium
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEMENTE_ALEATORIA = 42  # fixa para reprodutibilidade dos experimentos (importante para o artigo)
random.seed(SEMENTE_ALEATORIA)
np.random.seed(SEMENTE_ALEATORIA)

RAIO_TERRA_KM = 6371.0
VELOCIDADE_EMBARCACAO_KMH = 25.0

ACOES_EMBARCACAO = {
    "N": (1.0, 0.0),
    "S": (-1.0, 0.0),
    "L": (0.0, 1.0),
    "O": (0.0, -1.0),
    "NE": (0.7071, 0.7071),
    "NO": (0.7071, -0.7071),
    "SE": (-0.7071, 0.7071),
    "SO": (-0.7071, -0.7071),
    "PARADO": (0.0, 0.0),  # permite a embarcacao manter posicao quando ja esta proxima do alvo
}


# ---------------------------------------------------------------------------
# 1. Ingestao e engenharia de atributos
# ---------------------------------------------------------------------------

def carregar_e_processar_dados(caminho_arquivo):
    print(">>> 1. Ingerindo dados do Copernicus NetCDF...")
    ds = xr.open_dataset(caminho_arquivo)
    df = ds.to_dataframe().reset_index().dropna()

    print(">>> 2. Calculando grandezas fisicas vetoriais...")
    df["velocidade_ms"] = np.sqrt(df["uo"] ** 2 + df["vo"] ** 2)
    df["velocidade_kmh"] = df["velocidade_ms"] * 3.6
    df["direcao_graus"] = (np.degrees(np.arctan2(df["vo"], df["uo"])) + 360) % 360

    # Correcao de circularidade: 359 graus e 1 grau sao quase a mesma direcao,
    # mas em distancia Euclidiana ficam nos extremos opostos. Decompor em
    # seno/cosseno resolve isso antes do clustering.
    df["dir_sin"] = np.sin(np.radians(df["direcao_graus"]))
    df["dir_cos"] = np.cos(np.radians(df["direcao_graus"]))

    print(f"Registros validos no oceano: {len(df)}")
    return ds, df


# ---------------------------------------------------------------------------
# 2. Clustering nao supervisionado (DBSCAN) com justificativa de eps
# ---------------------------------------------------------------------------

def gerar_grafico_k_distancia(features_scaled, k=6, arquivo_saida="k_distance_plot.png"):
    print(f"\n>>> 3a. Gerando grafico k-distancia (k={k}) para justificar o eps do DBSCAN...")
    nbrs = NearestNeighbors(n_neighbors=k).fit(features_scaled)
    distancias, _ = nbrs.kneighbors(features_scaled)
    distancias_k = np.sort(distancias[:, k - 1])

    plt.figure(figsize=(8, 5))
    plt.plot(distancias_k)
    plt.xlabel("Pontos ordenados por distancia")
    plt.ylabel(f"Distancia ao {k}-esimo vizinho mais proximo")
    plt.title("Grafico k-distancia - justificativa do eps (DBSCAN)")
    plt.grid(alpha=0.3)
    plt.savefig(arquivo_saida, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"-> Grafico salvo em {arquivo_saida}. O 'cotovelo' da curva indica um eps razoavel.")
    return distancias_k


def aplicar_ia_clustering(df, eps=0.45, min_samples=6, n_amostra=800):
    print("\n>>> 3. Executando DBSCAN (Clustering Nao Supervisionado)...")
    df_amostra = df.sample(n=min(n_amostra, len(df)), random_state=42).copy()

    # dir_sin/dir_cos no lugar de direcao_graus bruta (corrige circularidade)
    features = df_amostra[["latitude", "longitude", "velocidade_kmh", "dir_sin", "dir_cos"]]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    gerar_grafico_k_distancia(features_scaled, k=min_samples)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df_amostra["cluster"] = dbscan.fit_predict(features_scaled)

    labels = df_amostra["cluster"]
    n_clusters = len(set(labels)) - (1 if -1 in labels.values else 0)
    n_anomalias = int((labels == -1).sum())

    print(f"-> Clusters identificados: {n_clusters}")
    print(f"-> Anomalias/ruidos detectados: {n_anomalias}")

    return df_amostra


# ---------------------------------------------------------------------------
# 3. Interpolacao espaco-temporal correta + simulacao de deriva com incerteza
# ---------------------------------------------------------------------------

def obter_corrente_interpolada(ds, lat, lon, tempo_alvo):
    """Interpola uo/vo em latitude, longitude E tempo simultaneamente.

    Correcao do bug original: a versao anterior buscava o vizinho mais
    proximo so em lat/lon, ignorando que o dataset tem 6 dias empilhados
    como linhas -- podia pegar corrente de um dia errado. Usar
    ds.interp() com os tres eixos resolve isso e tambem retorna NaN quando
    o ponto sai do dominio coberto pelos dados (em vez de "grudar" na borda).
    """
    ponto = ds.interp(
        latitude=lat,
        longitude=lon,
        time=tempo_alvo,
        method="linear",
        kwargs={"fill_value": np.nan},
    )
    uo = float(ponto["uo"].isel(depth=0).values)
    vo = float(ponto["vo"].isel(depth=0).values)
    return uo, vo


def simular_trajetoria_deriva(ds, lat_origem, lon_origem, total_horas=24,
                               passo_minutos=30, ruido_std=0.0):
    """Simula deriva lagrangeana. ruido_std > 0 perturba a corrente em cada
    passo (usado para gerar o ensemble Monte Carlo de incerteza)."""
    trajetoria = [(lat_origem, lon_origem, 0.0)]
    lat_atual, lon_atual = lat_origem, lon_origem
    dt_segundos = passo_minutos * 60
    tempo_inicial = ds.time.values[0]
    n_passos = int((total_horas * 60) / passo_minutos)

    for passo in range(1, n_passos + 1):
        tempo_decorrido_h = (passo - 1) * (passo_minutos / 60.0)
        tempo_alvo = tempo_inicial + np.timedelta64(int(tempo_decorrido_h * 3600), "s")

        try:
            uo, vo = obter_corrente_interpolada(ds, lat_atual, lon_atual, tempo_alvo)
        except Exception:
            break

        if np.isnan(uo) or np.isnan(vo):
            # trajetoria saiu do dominio coberto pelos dados (lat/lon/tempo)
            break

        if ruido_std > 0:
            uo += np.random.normal(0, ruido_std * (abs(uo) + 0.02))
            vo += np.random.normal(0, ruido_std * (abs(vo) + 0.02))

        d_lat = (vo * dt_segundos) / 111320.0
        d_lon = (uo * dt_segundos) / (111320.0 * np.cos(np.radians(lat_atual)))

        lat_atual += d_lat
        lon_atual += d_lon
        tempo_atual_h = passo * (passo_minutos / 60.0)
        trajetoria.append((lat_atual, lon_atual, tempo_atual_h))

    return trajetoria


def gerar_ensemble_monte_carlo(ds, lat_origem, lon_origem, total_horas=24,
                                passo_minutos=30, n_membros=30, ruido_std=0.15):
    print(f"\n>>> 4. Gerando ensemble Monte Carlo ({n_membros} membros) para "
          f"quantificar incerteza da deriva...")
    trajetoria_central = simular_trajetoria_deriva(
        ds, lat_origem, lon_origem, total_horas, passo_minutos, ruido_std=0.0
    )
    trajetorias = [trajetoria_central]
    for _ in range(n_membros - 1):
        traj = simular_trajetoria_deriva(
            ds, lat_origem, lon_origem, total_horas, passo_minutos, ruido_std=ruido_std
        )
        trajetorias.append(traj)

    print(f"-> {len(trajetorias)} trajetorias simuladas (1 central + "
          f"{len(trajetorias) - 1} perturbadas).")
    return trajetorias


# ---------------------------------------------------------------------------
# 4. Monte Carlo Tree Search - rota de interceptacao da embarcacao
# ---------------------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * RAIO_TERRA_KM * math.asin(math.sqrt(a))


def mover_embarcacao(lat, lon, acao, passo_horas):
    d_lat_unit, d_lon_unit = ACOES_EMBARCACAO[acao]
    dist_km = VELOCIDADE_EMBARCACAO_KMH * passo_horas
    d_lat = (d_lat_unit * dist_km) / 111.32
    d_lon = (d_lon_unit * dist_km) / (111.32 * math.cos(math.radians(lat)))
    return lat + d_lat, lon + d_lon


class NoMCTS:
    def __init__(self, estado, pai=None, acao_que_originou=None):
        self.estado = estado  # (lat, lon, passo)
        self.pai = pai
        self.acao_que_originou = acao_que_originou
        self.filhos = []
        self.visitas = 0
        self.valor_total = 0.0
        self.acoes_nao_expandidas = list(ACOES_EMBARCACAO.keys())

    def totalmente_expandido(self):
        return len(self.acoes_nao_expandidas) == 0

    def melhor_filho_uct(self, c=1.4):
        melhor, melhor_score = None, -float("inf")
        for filho in self.filhos:
            exploracao = c * math.sqrt(math.log(self.visitas) / filho.visitas)
            score = (filho.valor_total / filho.visitas) + exploracao
            if score > melhor_score:
                melhor, melhor_score = filho, score
        return melhor


PESO_PENALIDADE_GIRO = 8.0  # km "equivalentes" penalizados por mudanca de direcao
# (recalibrado: a embarcacao percorre ~25km/h * passo_horas por passo, entao a
# penalidade precisa ser da mesma ordem de grandeza para de fato desencorajar
# ziguezague, em vez de ser ofuscada pela distancia final)


def _caminho_de_acoes(no):
    """Reconstroi a sequencia de acoes da raiz ate o no (para saber qual foi
    a ultima acao antes do rollout comecar, e assim penalizar giros na
    transicao arvore -> rollout tambem, nao so dentro do rollout)."""
    acoes = []
    atual = no
    while atual.pai is not None:
        acoes.append(atual.acao_que_originou)
        atual = atual.pai
    acoes.reverse()
    return acoes


def _posicao_na_trajetoria_em_tempo(trajetoria, tempo_alvo_horas):
    """Retorna a posicao (lat, lon) da trajetoria mais proxima do tempo_alvo,
    em horas reais -- em vez de tratar o indice da lista como se fosse o
    tempo (bug corrigido: a trajetoria do poluente registra um ponto a cada
    30 min, mas a embarcacao avanca em passos de 1h; usar o contador de
    passos da embarcacao como indice direto da lista fazia o alvo ficar
    sempre 2x mais "no passado" do que deveria)."""
    if tempo_alvo_horas >= trajetoria[-1][2]:
        return trajetoria[-1][0], trajetoria[-1][1]
    ponto_mais_proximo = min(trajetoria, key=lambda p: abs(p[2] - tempo_alvo_horas))
    return ponto_mais_proximo[0], ponto_mais_proximo[1]


def _rollout(estado, ultima_acao, horizonte_restante, passo_horas, trajetorias_ensemble):
    lat, lon, passo = estado
    penalidade_giro = 0

    for _ in range(horizonte_restante):
        acao = random.choice(list(ACOES_EMBARCACAO.keys()))
        if ultima_acao is not None and acao != ultima_acao:
            penalidade_giro += 1
        ultima_acao = acao
        lat, lon = mover_embarcacao(lat, lon, acao, passo_horas)
        passo += 1

    # A incerteza sobre onde o poluente estara e representada sorteando um
    # membro do ensemble Monte Carlo a cada rollout -- isso faz o MCTS
    # otimizar a rota considerando o cone de incerteza, nao um ponto fixo.
    traj_amostrada = random.choice(trajetorias_ensemble)
    tempo_alvo_horas = passo * passo_horas
    lat_alvo, lon_alvo = _posicao_na_trajetoria_em_tempo(traj_amostrada, tempo_alvo_horas)
    distancia = haversine_km(lat, lon, lat_alvo, lon_alvo)

    # Penaliza rotas com muitas mudancas bruscas de direcao: uma embarcacao
    # real gasta combustivel e tempo manobrando, entao preferimos rotas mais
    # diretas quando a distancia final resultante for parecida.
    return -distancia - PESO_PENALIDADE_GIRO * penalidade_giro


def mcts_buscar(estado_inicial, horizonte, passo_horas, trajetorias_ensemble,
                 n_simulacoes=3000, verbose=True):
    if verbose:
        print(f"\n>>> 5. Executando MCTS ({n_simulacoes} simulacoes, horizonte de "
              f"{horizonte} passos) para planejar a rota de interceptacao...")
    raiz = NoMCTS(estado_inicial)

    for _ in range(n_simulacoes):
        no = raiz
        profundidade = 0

        # Selecao (UCT)
        while no.totalmente_expandido() and no.filhos and profundidade < horizonte:
            no = no.melhor_filho_uct()
            profundidade += 1

        # Expansao
        if profundidade < horizonte and no.acoes_nao_expandidas:
            acao = no.acoes_nao_expandidas.pop(
                random.randrange(len(no.acoes_nao_expandidas))
            )
            nova_lat, nova_lon = mover_embarcacao(no.estado[0], no.estado[1], acao, passo_horas)
            novo_estado = (nova_lat, nova_lon, no.estado[2] + 1)
            filho = NoMCTS(novo_estado, pai=no, acao_que_originou=acao)
            no.filhos.append(filho)
            no = filho
            profundidade += 1

        # Simulacao (rollout aleatorio), continuando a partir da ultima acao
        # tomada no caminho da arvore (para a penalidade de giro fazer sentido
        # tambem na transicao entre arvore e rollout)
        caminho = _caminho_de_acoes(no)
        ultima_acao = caminho[-1] if caminho else None
        recompensa = _rollout(no.estado, ultima_acao, horizonte - profundidade,
                               passo_horas, trajetorias_ensemble)

        # Retropropagacao
        no_atual = no
        while no_atual is not None:
            no_atual.visitas += 1
            no_atual.valor_total += recompensa
            no_atual = no_atual.pai

    melhor_rota = [raiz.estado]
    no = raiz
    while no.filhos:
        no = max(no.filhos, key=lambda f: f.visitas)
        melhor_rota.append(no.estado)

    if verbose:
        print(f"-> Rota de interceptacao com {len(melhor_rota) - 1} passos definida.")
    return melhor_rota


def planejar_rota_horizonte_retratil(estado_inicial, horizonte_total, horizonte_planejamento,
                                      passo_horas, trajetorias_ensemble, n_simulacoes=1500):
    """Em vez de planejar os horizonte_total passos de uma vez so (o que gera
    rotas mais 'tortas', pois o algoritmo se compromete cedo demais com um
    plano longo baseado em rollouts ruidosos), replaneja a cada passo:
    roda o MCTS com um horizonte curto (horizonte_planejamento), executa
    so a primeira acao recomendada, atualiza o estado, e roda de novo.
    Isso e o equivalente ao "receding horizon control" usado em sistemas de
    navegacao reais, e produz trajetorias mais suaves e mais realistas.
    """
    print(f"\n>>> 5. Planejando rota de interceptacao via MCTS com horizonte "
          f"retratil ({horizonte_total} passos totais, replanejando a cada "
          f"passo com horizonte de {horizonte_planejamento})...")

    rota_completa = [estado_inicial]
    estado_atual = estado_inicial

    for passo_executado in range(horizonte_total):
        horizonte_deste_replano = min(horizonte_planejamento, horizonte_total - passo_executado)
        plano_parcial = mcts_buscar(
            estado_atual,
            horizonte=horizonte_deste_replano,
            passo_horas=passo_horas,
            trajetorias_ensemble=trajetorias_ensemble,
            n_simulacoes=n_simulacoes,
            verbose=False,
        )
        proximo_estado = plano_parcial[1]  # so a 1a acao do (re)plano e de fato executada
        rota_completa.append(proximo_estado)
        estado_atual = proximo_estado

    print(f"-> Rota final com {len(rota_completa) - 1} passos definida "
          f"({horizonte_total} replanejamentos executados).")
    return rota_completa


# ---------------------------------------------------------------------------
# 5. Visualizacao integrada
# ---------------------------------------------------------------------------

def gerar_mapa_integrado(df_amostra, trajetorias_ensemble, rota_embarcacao,
                          horizonte_embarcacao_horas=None, arquivo_saida="mapa_clusters.html"):
    print("\n>>> 6. Renderizando mapa geoespacial integrado...")
    lat_media = df_amostra["latitude"].mean()
    lon_media = df_amostra["longitude"].mean()

    mapa = folium.Map(location=[lat_media, lon_media], zoom_start=7, tiles="OpenStreetMap")

    clusters_unicos = sorted(int(c) for c in df_amostra["cluster"].unique() if c != -1)
    cmap = plt.colormaps["tab20"]
    cores_map = {cid: mcolors.to_hex(cmap(i % 20)) for i, cid in enumerate(clusters_unicos)}

    for _, row in df_amostra.iterrows():
        cluster_id = int(row["cluster"])
        if cluster_id == -1:
            cor, nome_tag = "#555555", "Anomalia / Ruido"
        else:
            cor, nome_tag = cores_map[cluster_id], f"Cluster {cluster_id}"

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=4, color=cor, fill=True, fill_color=cor, fill_opacity=0.75,
            popup=f"<b>IA:</b> {nome_tag}<br><b>Vel:</b> {row['velocidade_kmh']:.2f} km/h",
        ).add_to(mapa)

    # Ensemble Monte Carlo (trajetorias perturbadas, finas e translucidas)
    for traj in trajetorias_ensemble[1:]:
        coords = [(p[0], p[1]) for p in traj]
        folium.PolyLine(coords, color="#e67e22", weight=1, opacity=0.25).add_to(mapa)

    # Trajetoria central (media) em preto tracejado
    coords_central = [(p[0], p[1]) for p in trajetorias_ensemble[0]]
    folium.PolyLine(coords_central, color="#000000", weight=4, opacity=0.9,
                     dash_array="6", popup="Trajetoria central prevista").add_to(mapa)

    folium.Marker(
        location=[coords_central[0][0], coords_central[0][1]],
        icon=folium.Icon(color="red", icon="info-sign"),
        popup="Origem do incidente (0h)",
    ).add_to(mapa)
    folium.Marker(
        location=[coords_central[-1][0], coords_central[-1][1]],
        icon=folium.Icon(color="blue", icon="flag"),
        popup="Posicao central prevista (final do horizonte de 24h da simulacao)",
    ).add_to(mapa)

    # Marcador extra: onde o poluente deve estar no fim da JANELA DE OPERACAO
    # da embarcacao (que costuma ser menor que o horizonte total da
    # simulacao de deriva) -- e este ponto, nao o de 24h, que a embarcacao
    # de fato tenta interceptar.
    if horizonte_embarcacao_horas is not None:
        lat_alvo_real, lon_alvo_real = _posicao_na_trajetoria_em_tempo(
            trajetorias_ensemble[0], horizonte_embarcacao_horas
        )
        folium.Marker(
            location=[lat_alvo_real, lon_alvo_real],
            icon=folium.Icon(color="purple", icon="record"),
            popup=(f"Posicao esperada do poluente ao fim da operacao da "
                   f"embarcacao (+{horizonte_embarcacao_horas:.0f}h) -- "
                   f"e este o alvo real do MCTS, nao o da bandeira azul."),
        ).add_to(mapa)

    # Rota da embarcacao de interceptacao (MCTS), em verde
    coords_rota = [(p[0], p[1]) for p in rota_embarcacao]
    folium.PolyLine(coords_rota, color="#27ae60", weight=4, opacity=0.9,
                     popup="Rota de interceptacao (MCTS)").add_to(mapa)
    folium.Marker(
        location=[coords_rota[0][0], coords_rota[0][1]],
        icon=folium.Icon(color="green", icon="play"),
        popup="Base da embarcacao de contencao",
    ).add_to(mapa)
    folium.Marker(
        location=[coords_rota[-1][0], coords_rota[-1][1]],
        icon=folium.Icon(color="green", icon="ok-sign"),
        popup="Posicao final da embarcacao (MCTS)",
    ).add_to(mapa)

    mapa.save(arquivo_saida)
    print(f">>> Mapa atualizado com sucesso em: {arquivo_saida}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    caminho = "dados_maritimos.nc"
    ds, dados_oceano = carregar_e_processar_dados(caminho)
    dados_com_ia = aplicar_ia_clustering(dados_oceano)

    ponto_inicial_lat = -24.5
    ponto_inicial_lon = -45.5

    ensemble = gerar_ensemble_monte_carlo(
        ds, ponto_inicial_lat, ponto_inicial_lon,
        total_horas=24, passo_minutos=30, n_membros=30, ruido_std=0.15,
    )

    # Base da embarcacao de contencao: ponto proximo (ex: porto de referencia)
    base_embarcacao_lat = -23.9
    base_embarcacao_lon = -46.0
    estado_inicial_embarcacao = (base_embarcacao_lat, base_embarcacao_lon, 0)

    horizonte_total_passos = 10
    passo_horas_embarcacao = 1.0

    rota = planejar_rota_horizonte_retratil(
        estado_inicial_embarcacao,
        horizonte_total=horizonte_total_passos,       # 10 passos * 1h = 10h de operacao no total
        horizonte_planejamento=4,     # mas so "enxerga" 4h a frente a cada replanejamento
        passo_horas=passo_horas_embarcacao,
        trajetorias_ensemble=ensemble,
        n_simulacoes=1500,            # por replanejamento (10x menos replanos que antes, entao compensa)
    )

    gerar_mapa_integrado(
        dados_com_ia, ensemble, rota,
        horizonte_embarcacao_horas=horizonte_total_passos * passo_horas_embarcacao,
    )