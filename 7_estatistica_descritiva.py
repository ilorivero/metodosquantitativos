"""
Estatística Descritiva no PPG Dataset
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv("PPG_Dataset.csv")
    
    # Selecionamos a feature '1275' como exemplo representativo de amplitude temporal do PPG
    serie = df['1275']
    
    print("=== Estatística Descritiva da Feature '1275' ===")
    print(f"Média:           {serie.mean():.6f}")
    print(f"Mediana:         {serie.median():.6f}")
    print(f"Moda:            {serie.mode()[0]:.6f}")
    print(f"Amplitude Total: {serie.max() - serie.min():.6f}")
    print(f"Variância:       {serie.var():.6f}")
    print(f"Desvio Padrão:   {serie.std():.6f}")
    print(f"Desvio Médio Absoluto (MAD aproximado): {np.mean(np.abs(serie - np.mean(serie))):.6f}")
    
    # Correlação e Covariância entre a feature '100' e a '1275'
    print("\n=== Relações Multivariadas (Feature '100' vs '1275') ===")
    print("Covariância:")
    print(np.cov(df['100'], df['1275']))
    print("\nCorrelação:")
    print(np.corrcoef(df['100'], df['1275']))
    
    # 4. Geração dos Gráficos (Histograma e Boxplot)
    print("\nGerando gráficos de visualização descritiva...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.set_theme(style="whitegrid")
    
    # Histograma
    sns.histplot(data=df, x='1275', hue='Label', kde=True, palette={'Normal': '#2ecc71', 'MI': '#e74c3c'}, ax=axes[0])
    axes[0].set_title("Histograma da Feature '1275'")
    
    # Boxplot
    sns.boxplot(data=df, x='Label', y='1275', palette={'Normal': '#2ecc71', 'MI': '#e74c3c'}, ax=axes[1])
    axes[1].set_title("Boxplot por Classe")
    
    plt.tight_layout()
    plt.savefig("estatistica_descritiva_1275.png")
    print("Gráfico descritivo salvo como 'estatistica_descritiva_1275.png'.")

if __name__ == "__main__":
    main()
