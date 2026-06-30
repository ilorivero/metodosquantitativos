# Métodos Quantitativos e Estatística — PPG Dataset

Este subdiretório `/metodosquantitativos` reúne scripts Python independentes desenvolvidos para a aplicação prática de técnicas de estatística multivariada, álgebra linear computacional e modelagem preditiva sobre o **Photoplethysmography (PPG) Dataset** (`PPG_Dataset.csv`).

Cada algoritmo serve a um propósito específico de exploração, compressão ou classificação diagnóstica de sinais fisiológicos.

---

## 📂 Organização dos Scripts e Para Que Servem

### [1_matrizes.py](./1_matrizes.py) — Álgebra Linear Computacional
* **Para que serve:** Demonstra as operações computacionais matriciais essenciais para o processamento de sinais de alta dimensionalidade.
* **O que faz:** Realiza multiplicação de submatrizes ($A \cdot B$), transposição ($A^T$), cálculo de determinante, inversão ($A^{-1}$), verificação de posto linear (*matrix rank*), extração de autovalores/autovetores e o cálculo manual de matrizes de cofatores e adjuntas.

### [2_pca.py](./2_pca.py) — Componentes Principais (PCA)
* **Para que serve:** Redução de dimensionalidade não-supervisionada e eliminação de multicolinearidade.
* **O que faz:** Rotaciona ortogonalmente o espaço original de 2.000 features temporais do PPG para alinhar-se com os eixos de variância máxima (componentes principais), exibindo a variância individual, variância acumulada explicada e scores de projeção.

### [3_svd.py](./3_svd.py) — Decomposição em Valores Singulares (SVD)
* **Para que serve:** Fatoração matricial robusta e aproximação de baixo posto.
* **O que faz:** Decompõe a matriz de sinal nas componentes ortogonais $U$, $V^T$ e autovalores singulares $s$. Realiza a reconstrução exata da matriz e demonstra a simplificação de sinal por meio de uma aproximação de Rank-3 (descarte de ruído).

### [4_analise_fatorial.py](./4_analise_fatorial.py) — Variáveis Latentes
* **Para que serve:** Agrupamento e identificação de fatores ocultos não observáveis que geram a estrutura de correlação entre os batimentos.
* **O que faz:** Estima as cargas fatoriais (*loadings*) comuns de 2 fatores latentes aplicando a rotação ortogonal **Varimax** para maximizar a interpretabilidade das features temporais, calculando comunalidades ($h^2$) e variâncias específicas.

### [5_analise_discriminante.py](./5_analise_discriminante.py) — Classificadores Estatísticos
* **Para que serve:** Classificação diagnóstica supervisionada baseada em limites estatísticos do sinal.
* **O que faz:** Divide os dados em treino e teste, e ajusta modelos de **Análise Discriminante Linear (LDA)** (fronteira plana assumindo covariâncias homogêneas) e **Análise Discriminante Quadrática (QDA)** (fronteira curva sem assumir covariâncias homogêneas). Exibe acurácias, matriz de confusão e a taxa de erro aparente (APER).

### [6_regressao.py](./6_regressao.py) — Modelagem Preditiva
* **Para que serve:** Modelagem de relações matemáticas para estimar amplitudes futuras ou probabilidades de diagnóstico.
* **O que faz:** 
  - *Regressão Linear Simples:* Preve a amplitude temporal '500' com base na amplitude '100'.
  - *Regressão Linear Multivariada:* Preve a amplitude '500' a partir das amplitudes '100', '200' e '300'.
  - *Regressão Logística:* Classifica de forma probabilística binária o diagnóstico ("Normal" vs "MI") com base no sinal, calculando coeficientes e acurácia.

### [7_estatistica_descritiva.py](./7_estatistica_descritiva.py) — EDA e Distribuição
* **Para que serve:** Análise exploratória inicial (EDA) para resumir os dados e visualizar densidades.
* **O que faz:** Calcula média, mediana, moda, amplitude, variância, desvio padrão e o desvio médio absoluto (MAD) da amplitude na feature selecionada `'1275'`. Avalia a correlação e covariância linear da feature '100' com a '1275', e salva os gráficos de Histograma de densidade e Boxplot como `estatistica_descritiva_1275.png`.

### [8_anova.py](./8_anova.py) — Teste de Médias (ANOVA)
* **Para que serve:** Validação de hipóteses estatísticas comparando a média de múltiplos grupos.
* **O que faz:** Divide as amostras de pacientes em três tercis de amplitude (Baixo, Médio e Alto) e executa uma **ANOVA de uma via (One-Way ANOVA)** na feature `'1000'` para validar se há diferença estatisticamente significativa entre as médias dos grupos estudados, reportando estatística F e valor-p.

### [9_series_temporais.py](./9_series_temporais.py) — Processamento Temporal
* **Para que serve:** Análise especializada de tendências e dinâmicas contínuas das séries temporais de PPG.
* **O que faz:** Realiza o processamento, visualização ou modelagem contínua das amplitudes fisiológicas sequenciais ao longo dos instantes temporais.

---

## 🚀 Como Executar os Scripts

### Pré-requisitos
Instale as dependências exigidas para execução dos algoritmos e geração de plots:

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn
```

### Execução dos Algoritmos
Todos os scripts assumem que o dataset `PPG_Dataset.csv` está localizado na pasta raiz do projeto (uma pasta acima deste subdiretório). Garanta que está executando a partir da pasta raiz ou diretamente de dentro da pasta `metodosquantitativos`:

```bash
# Estando dentro do diretório /metodosquantitativos:
python 1_matrizes.py
python 2_pca.py
python 3_svd.py
python 4_analise_fatorial.py
python 5_analise_discriminante.py
python 6_regressao.py
python 7_estatistica_descritiva.py
python 8_anova.py
python 9_series_temporais.py
```
