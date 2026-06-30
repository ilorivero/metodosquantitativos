"""
Análise de Componentes Principais (PCA) no PPG Dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

def main():
    df = pd.read_csv("PPG_Dataset.csv")
    X = df.drop(columns=["Label"]).to_numpy()
    
    # Padronização (necessária para PCA)
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA para obter os primeiros componentes
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    
    print("Variância explicada por componente:")
    print(pca.explained_variance_ratio_)
    
    print("\nVariância acumulada explicada:")
    print(np.cumsum(pca.explained_variance_ratio_))
    
    print("\nComponentes principais (autovetores / loadings) - Primeiros 2 componentes (primeras 5 features):")
    print(pca.components_[:2, :5])
    
    print("\nScores dos componentes (projeções) - Primeiros 5 registros:")
    print(X_pca[:5])

if __name__ == "__main__":
    main()
