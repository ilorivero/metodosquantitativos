"""
Análise Discriminante (LDA e QDA) no PPG Dataset
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    df = pd.read_csv("PPG_Dataset.csv")
    # Para LDA/QDA usaremos uma seleção de 20 colunas para viabilizar o cálculo do QDA sem singularidades
    X = df.iloc[:, 100:120].to_numpy()
    y = df["Label"].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # --- 1. Linear Discriminant Analysis (LDA) ---
    print("=== 1. Linear Discriminant Analysis (LDA) ===")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_scaled, y_train)
    
    y_pred_lda = lda.predict(X_test_scaled)
    acc_lda = accuracy_score(y_test, y_pred_lda)
    aper_lda = 1.0 - acc_lda
    
    print("Médias das classes no espaço original (primeiras 3 features selecionadas):")
    print(lda.means_[:, :3])
    print(f"\nAcurácia LDA: {acc_lda * 100:.2f}%")
    print(f"Taxa de Erro Aparente (APER) LDA: {aper_lda * 100:.2f}%")
    print("Matriz de Confusão LDA:")
    print(confusion_matrix(y_test, y_pred_lda))
    
    # --- 2. Quadratic Discriminant Analysis (QDA) ---
    print("\n=== 2. Quadratic Discriminant Analysis (QDA) ===")
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train_scaled, y_train)
    
    y_pred_qda = qda.predict(X_test_scaled)
    acc_qda = accuracy_score(y_test, y_pred_qda)
    aper_qda = 1.0 - qda.score(X_test_scaled, y_test)
    
    print(f"Acurácia QDA: {acc_qda * 100:.2f}%")
    print(f"Taxa de Erro Aparente (APER) QDA: {aper_qda * 100:.2f}%")
    print("Matriz de Confusão QDA:")
    print(confusion_matrix(y_test, y_pred_qda))

if __name__ == "__main__":
    main()
