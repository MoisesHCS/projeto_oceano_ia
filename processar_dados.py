import folium
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def carregar_e_processar_dados(caminho_arquivo):
    print(">>> 1. Ingerindo dados do Copernicus NetCDF...")
    ds = xr.open_dataset(caminho_arquivo)
    df = ds.to_dataframe().reset_index().dropna()

    print(">>> 2. Calculando grandezas físicas vetoriais...")
    df['velocidade_ms'] = np.sqrt(df['uo'] ** 2 + df['vo'] ** 2)
    df['velocidade_kmh'] = df['velocidade_ms'] * 3.6
    df['direcao_graus'] = (np.degrees(np.arctan2(df['vo'], df['uo'])) + 360) % 360

    print(f"Registros válidos no oceano: {len(df)}")
    return df


def aplicar_ia_clustering(df):
    print("\n>>> 3. Executando DBSCAN (Clustering Não Supervisionado)...")
    df_amostra = df.sample(n=min(800, len(df)), random_state=42).copy()

    features = df_amostra[['latitude', 'longitude', 'velocidade_kmh', 'direcao_graus']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    dbscan = DBSCAN(eps=0.45, min_samples=6)
    df_amostra['cluster'] = dbscan.fit_predict(features_scaled)

    labels = df_amostra['cluster']
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_anomalias = list(labels).count(-1)

    print(f"-> Clusters identificados: {n_clusters}")
    print(f"-> Anomalias/Ruídos detectados: {n_anomalias}")

    return df_amostra


def simular_trajetoria_deriva(df_completo, lat_origem, lon_origem, total_horas=24, passo_minutos=30):
    print(f"\n>>> 4. Simulando trajetória de dispersão de poluente ({total_horas}h)...")

    trajetoria = [(lat_origem, lon_origem, 0)]
    lat_atual, lon_atual = lat_origem, lon_origem
    dt_segundos = passo_minutos * 60

    for passo in range(1, int((total_horas * 60) / passo_minutos) + 1):
        distancias = (df_completo['latitude'] - lat_atual) ** 2 + (df_completo['longitude'] - lon_atual) ** 2
        vizinho_mais_proximo = df_completo.loc[distancias.idxmin()]

        uo = vizinho_mais_proximo['uo']
        vo = vizinho_mais_proximo['vo']

        d_lat = (vo * dt_segundos) / 111320.0
        d_lon = (uo * dt_segundos) / (111320.0 * np.cos(np.radians(lat_atual)))

        lat_atual += d_lat
        lon_atual += d_lon
        tempo_atual = (passo * passo_minutos) / 60.0

        trajetoria.append((lat_atual, lon_atual, tempo_atual))

    return trajetoria


def gerar_mapa_integrado(df_amostra, trajetoria, arquivo_saida="mapa_clusters.html"):
    print("\n>>> 5. Renderizando mapa geoespacial integrado...")
    lat_media = df_amostra['latitude'].mean()
    lon_media = df_amostra['longitude'].mean()

    mapa = folium.Map(location=[lat_media, lon_media], zoom_start=7, tiles='OpenStreetMap')
    paleta_cores = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

    # 1. Plota os Clusters do DBSCAN
    for _, row in df_amostra.iterrows():
        cluster_id = int(row['cluster'])
        cor = '#808080' if cluster_id == -1 else paleta_cores[cluster_id % len(paleta_cores)]
        nome_tag = "Anomalia / Vórtice" if cluster_id == -1 else f"Cluster {cluster_id}"

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3.5,
            color=cor,
            fill=True,
            fill_opacity=0.6,
            popup=f"<b>IA:</b> {nome_tag}<br><b>Vel:</b> {row['velocidade_kmh']:.2f} km/h"
        ).add_to(mapa)

    # 2. Plota a Trajetória da Mancha (Linha tracejada preta para contraste)
    coords_linha = [(p[0], p[1]) for p in trajetoria]
    folium.PolyLine(
        coords_linha,
        color='#000000',
        weight=5,
        opacity=0.9,
        dash_array='8',
        popup="Trajetória Prevista de Dispersão (24h)"
    ).add_to(mapa)

    # Marcador inicial (Origem)
    folium.Marker(
        location=[trajetoria[0][0], trajetoria[0][1]],
        icon=folium.Icon(color='red', icon='warning-sign'),
        popup=f"<b>Origem do Incidente (0h)</b><br>Lat: {trajetoria[0][0]:.2f}, Lon: {trajetoria[0][1]:.2f}"
    ).add_to(mapa)

    # Marcador final (+24h)
    folium.Marker(
        location=[trajetoria[-1][0], trajetoria[-1][1]],
        icon=folium.Icon(color='blue', icon='flag'),
        popup=f"<b>Posição Prevista (+24h)</b><br>Lat: {trajetoria[-1][0]:.2f}, Lon: {trajetoria[-1][1]:.2f}"
    ).add_to(mapa)

    mapa.save(arquivo_saida)
    print(f">>> Mapa atualizado com sucesso em: {arquivo_saida}")


if __name__ == "__main__":
    caminho = "dados_maritimos.nc"
    dados_oceano = carregar_e_processar_dados(caminho)
    dados_com_ia = aplicar_ia_clustering(dados_oceano)

    ponto_inicial_lat = -24.5
    ponto_inicial_lon = -45.5

    rota_poluente = simular_trajetoria_deriva(dados_oceano, ponto_inicial_lat, ponto_inicial_lon, total_horas=24)
    gerar_mapa_integrado(dados_com_ia, rota_poluente)