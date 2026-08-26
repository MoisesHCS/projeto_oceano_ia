import folium
import numpy as np
import pandas as pd
import xarray as xr


def processar_dados_oceano(caminho_arquivo):
    print(">>> 1. Carregando dados oceanográficos...")
    ds = xr.open_dataset(caminho_arquivo)
    df = ds.to_dataframe().reset_index().dropna()

    print(">>> 2. Calculando magnitude e direção das correntes...")
    # Velocidade resultante (m/s) pela norma euclidiana: sqrt(uo^2 + vo^2)
    df['velocidade_ms'] = np.sqrt(df['uo'] ** 2 + df['vo'] ** 2)
    df['velocidade_kmh'] = df['velocidade_ms'] * 3.6

    # Direção do vetor em graus (0° a 360°)
    df['direcao_graus'] = (np.degrees(np.arctan2(df['vo'], df['uo'])) + 360) % 360

    print(f"Total de pontos processados: {len(df)}")
    print(df[['latitude', 'longitude', 'uo', 'vo', 'velocidade_kmh', 'direcao_graus']].head())

    return df


def gerar_mapa_correntes(df, arquivo_saida="mapa_correntes.html"):
    print("\n>>> 3. Gerando mapa interativo...")
    # Centraliza o mapa na média das coordenadas
    lat_media = df['latitude'].mean()
    lon_media = df['longitude'].mean()

    mapa = folium.Map(location=[lat_media, lon_media], zoom_start=8, tiles='CartoDB dark_matter')

    # Plota uma amostra dos pontos com cores baseadas na velocidade da corrente
    amostra = df.sample(n=min(300, len(df)), random_state=42)

    for _, row in amostra.iterrows():
        # Define cor de acordo com a intensidade da corrente
        cor = 'blue' if row['velocidade_kmh'] < 1.0 else ('orange' if row['velocidade_kmh'] < 2.0 else 'red')

        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=4,
            color=cor,
            fill=True,
            fill_opacity=0.7,
            popup=(
                f"<b>Lat/Lon:</b> {row['latitude']:.2f}, {row['longitude']:.2f}<br>"
                f"<b>Velocidade:</b> {row['velocidade_kmh']:.2f} km/h<br>"
                f"<b>Direção:</b> {row['direcao_graus']:.1f}°"
            )
        ).add_to(mapa)

    mapa.save(arquivo_saida)
    print(f">>> Mapa salvo com sucesso em: {arquivo_saida}")


if __name__ == "__main__":
    caminho = "dados_maritimos.nc"
    tabela_final = processar_dados_oceano(caminho)
    gerar_mapa_correntes(tabela_final)