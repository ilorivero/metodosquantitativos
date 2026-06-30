"""
Álgebra Linear e Operações Matriciais no PPG Dataset
"""
import pandas as pd
import numpy as np

def main():
    # Busca o CSV localmente ou no diretório pai
    df = pd.read_csv("PPG_Dataset.csv")
    X = df.drop(columns=["Label"]).to_numpy()
    
    # Submatriz 3x3 para demonstração de operações algébricas básicas
    A = X[:3, :3]
    B = X[3:6, :3]
    
    print("Matriz A (3x3):")
    print(A)
    print("\nMatriz B (3x3):")
    print(B)
    
    # Multiplicação
    C = np.dot(A, B)
    print("\nProduto A * B (np.dot):")
    print(C)
    
    # Transposta
    print("\nTransposta de A (A.T):")
    print(A.T)
    
    # Inversa e Determinante
    det_A = np.linalg.det(A)
    print(f"\nDeterminante de A: {det_A:.6f}")
    if np.abs(det_A) > 1e-9:
        print("\nInversa de A:")
        print(np.linalg.inv(A))
        
    # Posto (Rank)
    posto = np.linalg.matrix_rank(A)
    print(f"\nPosto da matriz A: {posto}")
    
    # Autovalores e Autovetores
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print("\nAutovalores de A:")
    print(eigenvalues)
    print("\nAutovetores de A:")
    print(eigenvectors)
    
    # Cofatores e Adjunta (3x3)
    MC = np.zeros((3,3))
    idx = np.array(range(3))
    for i in range(3):
        for j in range(3):
            fidx = idx[idx != i]
            cidx = idx[idx != j]
            # Extrai submatriz 2x2
            cof = A[fidx[:, None], cidx]
            MC[i,j] = ((-1)**(i+j)) * np.linalg.det(cof)
            
    print("\nMatriz de Cofatores de A:")
    print(MC)
    print("\nMatriz Adjunta de A (Cofatores Transposta):")
    print(MC.T)

if __name__ == "__main__":
    main()
