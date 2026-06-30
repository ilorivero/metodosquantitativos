"""
Decomposição em Valores Singulares (SVD) no PPG Dataset
"""
import pandas as pd
import numpy as np
import os

def main():
    df = pd.read_csv("PPG_Dataset.csv")
    X = df.drop(columns=["Label"]).to_numpy()
    
    # Seleciona uma pequena submatriz (ex: 6x6) para clareza na demonstração
    A = X[:6, :6]
    
    print("Matriz original A (6x6):")
    print(A)
    
    # Aplicação do SVD
    U, s, Vt = np.linalg.svd(A)
    
    print("\nMatriz U (vetores singulares à esquerda):")
    print(np.round(U, 4))
    
    print("\nValores Singulares (vetor s):")
    print(np.round(s, 4))
    
    print("\nMatriz V^T (vetores singulares à direita transpostos):")
    print(np.round(Vt, 4))
    
    # Reconstrução da matriz original usando os valores singulares
    S = np.diag(s)
    A_reconstructed = U @ S @ Vt
    print("\nMatriz Reconstruída (U * S * V^T):")
    print(np.round(A_reconstructed, 4))
    
    # Redução de Dimensionalidade: Mantendo apenas os 3 primeiros valores singulares
    k = 3
    S_k = np.zeros((6, 6))
    S_k[:k, :k] = np.diag(s[:k])
    A_reduced = U @ S_k @ Vt
    print(f"\nMatriz com Redução de Dimensionalidade (Rank-{k}):")
    print(np.round(A_reduced, 4))

if __name__ == "__main__":
    main()
