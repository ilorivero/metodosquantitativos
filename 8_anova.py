"""
Análise de Variância (ANOVA) no PPG Dataset
"""
import pandas as pd
import numpy as np
import scipy.stats as stats

def main():
    df = pd.read_csv("PPG_Dataset.csv")
    
    # Criamos três categorias de pacientes baseado na amplitude média absoluta global dos sinais
    mean_amp = df.drop(columns=["Label"]).abs().mean(axis=1)
    
    # Dividindo em 3 grupos por tercis (T1: Baixo, T2: Médio, T3: Alto)
    tercis = pd.qcut(mean_amp, q=3, labels=["Baixo", "Médio", "Alto"])
    
    # Nossa variável dependente será a feature temporal '1000'
    g_baixo = df.loc[tercis == "Baixo", "1000"]
    g_medio = df.loc[tercis == "Médio", "1000"]
    g_alto = df.loc[tercis == "Alto", "1000"]
    
    print("=== Médias da Feature '1000' por Grupo ===")
    print(f"Grupo Baixo (N={len(g_baixo)}): Média = {g_baixo.mean():.6f}")
    print(f"Grupo Médio (N={len(g_medio)}): Média = {g_medio.mean():.6f}")
    print(f"Grupo Alto  (N={len(g_alto)}):  Média = {g_alto.mean():.6f}")
    
    # 2. Executando ANOVA One-way
    f_stat, p_val = stats.f_oneway(g_baixo, g_medio, g_alto)
    
    print("\n=== Resultados da ANOVA de Uma Via (One-Way ANOVA) ===")
    print(f"Estatística F: {f_stat:.6f}")
    print(f"Valor-p:       {p_val:.6e}")
    
    alpha = 0.05
    if p_val < alpha:
        print("\nDecisão: Rejeita-se a hipótese nula (H0).")
        print("Existe diferença estatisticamente significativa entre as médias dos grupos estudados.")
    else:
        print("\nDecisão: Não se rejeita a hipótese nula (H0).")
        print("Não há evidências de diferença estatisticamente significativa entre as médias dos grupos.")

if __name__ == "__main__":
    main()
