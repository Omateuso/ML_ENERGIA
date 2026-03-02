#%%
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
#%%
def gerar_graficos_comparativos(pasta_input='output/previsao'):
    # Procurando arquivos de previsão gerados na pasta output
    arquivos = glob.glob(os.path.join(pasta_input, 'previsao_2025_*.csv'))

    if not arquivos:
        print("Nenhum arquivo encontrado. Rode o predict_2025.py")
        return
    
    for arquivo in arquivos:
        # Extraindo o nome do subsistema do nome do arquivo
        subsistema = arquivo.split('_')[-1].replace('.csv', '')
        
        df = pd.read_csv(arquivo)
        if df.empty:
            print(f"Arquivo {arquivo} está vazio. Pulando...")
            continue

        # Convertendo para datetime para que o eixo X fique correto
        df['din_instante'] = pd.to_datetime(df['din_instante'])
        df = df.sort_values('din_instante')

        # Criando gráfico
        plt.figure(figsize=(12,6))

        plt.plot(df['din_instante'], df['val_cargaenergiamwmed'],
                 label='Carga Real', color='#1f77b4', alpha = 0.8)
        plt.plot(df['din_instante'], df['previsao_modelo'],
                 label='Previsão (XGBoost)', color='#d62728', linestyle='--')
        plt.title(f'Desempenho do modelo - Subsistema {subsistema} (2025)')
        plt.xlabel('Data')
        plt.ylabel('Carga (MWmed)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        # Salvando Gráfico
        nome_grafico = f'output/plots/plot_desempenho_{subsistema}.png'
        plt.savefig(nome_grafico)
        plt.close()

        print(f"Gráfico para o subsistema {subsistema} gerado com sucesso !")

if __name__ == "__main__":
    gerar_graficos_comparativos()

# %%
