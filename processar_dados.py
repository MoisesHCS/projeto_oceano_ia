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
    # Velocidade resultante (km/h) e ângulo azimutal (graus)
    df['velocidade_ms'] = np.sqrt(df['uo'] ** 2 + df['vo'] ** 2)
    df['velocidade_kmh'] = df['velocidade_ms'] * 3.6
    df['direcao_graus'] = (np.degrees(np.arctan2(df['vo'], df['uo'])) + 360) % 360

    print(f"Registros válidos no oceano: {len(df)}")
    return df


def aplicar_ia_clustering(df):
    print("\n>>> 3. Executando modelo de IA: DBSCAN (Clustering Não Supervisionado)...")

    # Amostra representativa para processamento e visualização
    df_amostra = df.sample(n=min(800, len(df)), random_state=42).copy()

    # Features oceanográficas selecionadas para a IA
    features = df_amostra[['latitude', 'longitude', 'velocidade_kmh', 'direcao_graus']]

    # Normalização dos atributos (essencial para cálculo de distância euclidiana)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Hiperparâmetros do DBSCAN (raio eps e densidade mínima de vizinhos)
    dbscan = DBSCAN(eps=0.45, min_samples=6)
    df_amostra['cluster'] = dbscan.fit_predict(features_scaled)

    # Estatísticas de agrupamento
    labels = df_amostra['cluster']
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_anomalias = list(labels).count(-1)

    print(f"-> Clusters identificados pela IA: {n_clusters}")
    print(f"-> Pontos classificados como anomalia/ruído (-1): {n_anomalias}")

    return df_amostra


def gerar_mapa_clusters(df_amostra, arquivo_saida="mapa_clusters.html"):
    print("\n>>> 4. Renderizando mapa geoespacial com as predições do modelo...")
    lat_media = df_amostra['latitude'].mean()
    lon_media = df_amostra['longitude'].mean()

    mapa = folium.Map(location=[lat_media, lon_media], zoom_start=7, tiles='CartoDB dark_matter')

    # Paleta de cores para os diferentes clusters
    paleta_cores = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6']

    for _, row in df_amostra.iterrows():
        cluster_id = int(row['cluster'])

        if cluster_id == -1:
            cor = '#808080'  # Cinza para ruídos/anomalias
            nome_tag = "Anomalia / Vórtice Isolado"
        else:
            cor = paleta_cores[cluster_id % len(paleta_cores)]
            nome_tag = f"Cluster {cluster_id}"

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=cor,
            fill=True,
            fill_opacity=0.85,
            popup=(
                f"<b>Classificação IA:</b> {nome_tag}<br>"
                f"<b>Velocidade:</b> {row['velocidade_kmh']:.2f} km/h<br>"
                f"<b>Direção:</b> {row['direcao_graus']:.1f}°<br>"
                f"<b>Lat/Lon:</b> {row['latitude']:.2f}, {row['longitude']:.2f}"
            )
        ).add_to(mapa)

    mapa.save(arquivo_saida)
    print(f">>> Mapa interativo salvo com sucesso em: {arquivo_saida}")


if __name__ == "__main__":
    caminho = "dados_maritimos.nc"
    dados_oceano = carregar_e_processar_dados(caminho)
    dados_com_ia = aplicar_ia_clustering(dados_oceano)
    gerar_mapa_clusters(dados_com_ia)