import matplotlib.pyplot as plt
import numpy as np
import json
import os

current_path = os.path.dirname(__file__)



def read_json_file(file_name):
    try:
        with open(file_name, 'r') as file:
            json_data = json.load(file)
        return json_data
    except FileNotFoundError:
        print(f"The file {file_name} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON in the file {file_name}. Check the format.")
        return None

json_file = read_json_file(os.path.join(current_path, "results.json"))

sets_fuzzy = ["CONJUNTO 5", "CONJUNTO 9", "CONJUNTO 11", "CONJUNTO 5 ANFIS", "CONJUNTO 9 ANFIS"]

for set_fuzzy in sets_fuzzy:
# Dados para 11 conjuntos Fuzzy
    print(set_fuzzy)
    features = json_file[set_fuzzy]["QTDE_FEATURES"]
    rmse = json_file[set_fuzzy]["RMSE_DADOS_TESTE"]
    num_rules = json_file[set_fuzzy]["QTDE_REGRAS_EXTRAIDAS"]

    # Criar figura e eixos
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Configurar eixo y da esquerda (RMSE)
    color = 'tab:red'
    ax1.set_xlabel('Number of Features', fontsize=14)
    ax1.set_ylabel('RMSE (cycles)', color=color, fontsize=14)
    ax1.plot(features, rmse, color=color, alpha=0.7, label='RMSE', marker='o', markersize=7, linestyle='-', linewidth=0.9)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim((17, 37))

    # Ajustar o tamanho da fonte dos valores dos eixos x
    plt.xticks(features, fontsize=12)
    plt.yticks(fontsize=12)

    # Adicionar números sobre os pontos de RMSE
    for i, value in enumerate(rmse):
        ax1.text(features[i], value + 0.3, round(value, 2), ha='center', va='bottom', color='black', fontsize=12)

    # Criar eixo y da direita (Quantidade de Regras)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Number of Rules', color=color, fontsize=14)
    ax2.plot(features, num_rules, color=color, alpha=0.7, label='Number of Rules', marker='o', markersize=7, linestyle='-', linewidth=0.9)
    ax2.tick_params(axis='y', labelcolor=color)

    # Ajustar o tamanho da fonte dos valores dos eixos x
    plt.xticks(features, fontsize=12)
    plt.yticks(fontsize=12)

    # Adicionar legenda
    fig.tight_layout()
    fig.legend(loc="upper center", fontsize=12)
    fig.savefig(os.path.join(current_path, f"{set_fuzzy}.svg"), format='svg')
    # Mostrar o gráfico
    plt.plot()

amount_features = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
rmse_test = [21.88, 20.59, 19.77, 19.44, 19.42, 19.28, 19.24, 19.15, 19.13, 19.09, 19.09, 19.07, 19.08, 19.08, 19.05, 19.04, 19.04, 19.04]
# Criar figura e eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Configurar eixo x e y
ax.set_xlabel('Number of Features', fontsize=14)
ax.set_ylabel('RMSE (cycles)', fontsize=14)

# Criar gráfico de barras
bars = ax.bar(amount_features, rmse_test, color='tab:red', alpha=0.7)
plt.xticks(amount_features, fontsize=12)
plt.yticks(fontsize=12)
ax.set_ylim((17, 22.5))
# # Adicionar números sobre as barras de dados
# for bar, value in zip(bars, rmse_test):
#     yval = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width() / 2, yval, round(value, 2), ha='center', va='bottom', color='black')

# Exibir o gráfico
plt.savefig(os.path.join(current_path, "light_gbm.svg"), format='svg')
