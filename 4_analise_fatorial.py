"""
Análise Fatorial no PPG Dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

def main():
    df = pd.read_csv("PPG_Dataset.csv")
    # Usando as primeiras 10 colunas para facilitar a visualização e interpretação das correlações
    features = df.columns[:10]
    X = df[features].to_numpy()
    
    # Padronização
    X_scaled = StandardScaler().fit_transform(X)
    
    # Análise Fatorial com 2 fatores e rotação Varimax
    fa = FactorAnalysis(n_components=2, rotation="varimax", random_state=42)
    X_fa = fa.fit_transform(X_scaled)
    
    # Cargas fatoriais (Factor Loadings)
    print("Cargas Fatoriais (Factor Loadings):")
    loadings = fa.components_.T
    df_loadings = pd.DataFrame(loadings, index=features, columns=["Fator 1", "Fator 2"])
    print(df_loadings)
    
    # Comunidades (h^2) e Variâncias Específicas
    # A variância específica é dada por noise_variance_
    specific_variance = fa.noise_variance_
    communality = 1.0 - specific_variance
    
    print("\nComunalidades (h^2) e Variância Específica (Residual):")
    df_comm = pd.DataFrame({
        "Comunalidade (h^2)": communality,
        "Variância Específica": specific_variance
    }, index=features)
    print(df_comm)
    
    print("\nScores Fatoriais (primeiras 5 amostras):")
    print(X_fa[:5])

if __name__ == "__main__":
    main()
