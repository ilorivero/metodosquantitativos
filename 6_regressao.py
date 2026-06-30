"""
Modelos de Regressão no PPG Dataset
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score

def main():
    df = pd.read_csv("PPG_Dataset.csv")
    X = df.drop(columns=["Label"]).to_numpy()
    y = df["Label"].map({"Normal": 0, "MI": 1}).to_numpy()
    
    # 1. Regressão Linear Simples
    # Relação entre a amplitude temporal '100' e a amplitude temporal '500'
    print("=== 1. Regressão Linear Simples ===")
    x_simple = X[:, [100]]
    y_simple = X[:, 500]
    
    lr_simple = LinearRegression()
    lr_simple.fit(x_simple, y_simple)
    y_pred_simple = lr_simple.predict(x_simple)
    
    print(f"Coeficiente Angular (Beta 1): {lr_simple.coef_[0]:.6f}")
    print(f"Intercepto (Beta 0):          {lr_simple.intercept_:.6f}")
    print(f"Coeficiente de Determinação R^2: {r2_score(y_simple, y_pred_simple):.4f}")
    
    # 2. Regressão Linear Multivariada
    # Prever a amplitude '500' com base nas amplitudes '100', '200' e '300'
    print("\n=== 2. Regressão Linear Multivariada ===")
    X_multi = X[:, [100, 200, 300]]
    
    lr_multi = LinearRegression()
    lr_multi.fit(X_multi, y_simple)
    y_pred_multi = lr_multi.predict(X_multi)
    
    print("Coeficientes (Beta 1, Beta 2, Beta 3):")
    print(lr_multi.coef_)
    print(f"Intercepto (Beta 0):                 {lr_multi.intercept_:.6f}")
    print(f"R^2 Multivariado:                    {r2_score(y_simple, y_pred_multi):.4f}")
    
    # 3. Regressão Logística
    # Prever a variável alvo binária (Label) com base nas features do PPG
    print("\n=== 3. Regressão Logística ===")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[:, :50]) # Usando as primeiras 50 features para demonstração rápida
    
    logit = LogisticRegression(max_iter=1000)
    logit.fit(X_scaled, y)
    y_pred_logit = logit.predict(X_scaled)
    
    print(f"Acurácia do Modelo: {accuracy_score(y, y_pred_logit)*100:.2f}%")
    print("Primeiros 5 coeficientes aprendidos pelo modelo:")
    print(logit.coef_[0][:5])

if __name__ == "__main__":
    main()
