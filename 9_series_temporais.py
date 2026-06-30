"""
Visualizacao de Series Temporais PPG: Normal vs Infarto (MI)
============================================================
Gera dois graficos complementares:

1. ENVELOPE TEMPORAL (mediana ± IQR por classe)
   Para cada um dos 2000 instantes temporais, calcula a mediana e o
   intervalo interquartil (25° e 75° percentil) de todos os pacientes
   de cada classe. A faixa sombreada mostra a dispersao tipica do sinal
   e onde as duas classes divergem.

2. CURVA DE F-STATISTIC AO LONGO DO TEMPO
   Para cada ponto temporal, aplica uma ANOVA one-way (Normal vs MI)
   e plota o valor F resultante. Picos de F indicam os instantes do
   sinal PPG mais discriminativos para o diagnostico de infarto.

Saida: ppg_visualizacao_temporal.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── 1. Carregar dados ────────────────────────────────────────────────────────
print("Carregando PPG_Dataset.csv...")
df = pd.read_csv("PPG_Dataset.csv")

labels = df["Label"]
X = df.drop(columns=["Label"]).to_numpy()
t = np.arange(X.shape[1])  # eixo temporal: 0 a 1999

mask_normal = (labels == "Normal").to_numpy()
mask_mi     = (labels == "MI").to_numpy()

X_normal = X[mask_normal]
X_mi     = X[mask_mi]

print(f"  Normal: {X_normal.shape[0]} pacientes")
print(f"  MI:     {X_mi.shape[0]} pacientes")

# ── 2. Calcular estatísticas por ponto temporal ──────────────────────────────
print("Calculando estatisticas temporais...")

def envelope(X_grupo):
    med  = np.median(X_grupo, axis=0)
    p25  = np.percentile(X_grupo, 25, axis=0)
    p75  = np.percentile(X_grupo, 75, axis=0)
    p5   = np.percentile(X_grupo, 5,  axis=0)
    p95  = np.percentile(X_grupo, 95, axis=0)
    return med, p25, p75, p5, p95

med_n, p25_n, p75_n, p5_n, p95_n = envelope(X_normal)
med_m, p25_m, p75_m, p5_m, p95_m = envelope(X_mi)

# F-statistic ponto a ponto
print("Calculando F-statistic para cada ponto temporal (isso leva alguns segundos)...")
f_vals = np.array([
    stats.f_oneway(X_normal[:, i], X_mi[:, i]).statistic
    for i in range(X.shape[1])
])

# ── 3. Figura ────────────────────────────────────────────────────────────────
print("Gerando figura...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor="#F8F9FA")
fig.suptitle(
    "Sinais PPG: Normal vs Infarto do Miocardio (MI)\n"
    "Envelope Temporal e Discriminabilidade por Ponto",
    fontsize=15, fontweight="bold", y=0.98, color="#1F3864"
)

COR_N  = "#2ECC71"   # verde  — Normal
COR_M  = "#E74C3C"   # vermelho — MI
COR_F  = "#8E44AD"   # roxo  — F-statistic

# ── Painel 1: Envelope ───────────────────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor("#FAFAFA")

# Faixa externa (5–95%)
ax1.fill_between(t, p5_n,  p95_n,  color=COR_N, alpha=0.10, label="_nolegend_")
ax1.fill_between(t, p5_m,  p95_m,  color=COR_M, alpha=0.10, label="_nolegend_")

# Faixa IQR (25–75%)
ax1.fill_between(t, p25_n, p75_n, color=COR_N, alpha=0.30, label="Normal IQR (25–75%)")
ax1.fill_between(t, p25_m, p75_m, color=COR_M, alpha=0.30, label="MI IQR (25–75%)")

# Medianas
ax1.plot(t, med_n, color=COR_N, linewidth=1.8, label="Normal — mediana")
ax1.plot(t, med_m, color=COR_M, linewidth=1.8, label="MI — mediana")

ax1.set_title("Envelope Temporal: Mediana ± IQR por Classe", fontsize=12, color="#333333")
ax1.set_ylabel("Amplitude do Sinal PPG", fontsize=11)
ax1.set_xlabel("Instante Temporal (t)", fontsize=11)
ax1.legend(loc="upper right", fontsize=10, framealpha=0.85)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.set_xlim(0, len(t) - 1)

# ── Painel 2: F-statistic ────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor("#FAFAFA")

ax2.plot(t, f_vals, color=COR_F, linewidth=1.2, alpha=0.85)
ax2.fill_between(t, 0, f_vals, color=COR_F, alpha=0.15)

# Linha de referência F critico (alpha=0.05, df1=1, df2=2574)
f_critico = stats.f.ppf(0.95, 1, X.shape[0] - 2)
ax2.axhline(f_critico, color="orange", linewidth=1.5, linestyle="--",
            label=f"F critico (alpha=0.05) = {f_critico:.2f}")

# Destacar os 5 picos mais discriminativos
top5_idx = np.argsort(f_vals)[-5:]
for idx in top5_idx:
    ax2.annotate(f"t={idx}", xy=(idx, f_vals[idx]),
                 xytext=(idx + 30, f_vals[idx] + f_vals.max() * 0.03),
                 fontsize=8, color="#1F3864",
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))

ax2.set_title("F-statistic por Ponto Temporal (Discriminabilidade Normal vs MI)", fontsize=12, color="#333333")
ax2.set_ylabel("Estatistica F (ANOVA)", fontsize=11)
ax2.set_xlabel("Instante Temporal (t)", fontsize=11)
ax2.legend(fontsize=10, framealpha=0.85)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.set_xlim(0, len(t) - 1)
ax2.set_ylim(bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.96])

output = "ppg_visualizacao_temporal.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
print(f"\nGrafico salvo como '{output}'")

# ── 4. Sumário dos pontos mais discriminativos ───────────────────────────────
top10 = np.argsort(f_vals)[-10:][::-1]
print("\nTop 10 pontos temporais mais discriminativos (maior F):")
print(f"{'Rank':>4}  {'t':>6}  {'F-stat':>12}  {'Media Normal':>14}  {'Media MI':>10}")
print("-" * 56)
for rank, idx in enumerate(top10, 1):
    print(f"{rank:>4}  {idx:>6}  {f_vals[idx]:>12.1f}  "
          f"{X_normal[:, idx].mean():>14.6f}  {X_mi[:, idx].mean():>10.6f}")
