import numpy as np
import matplotlib.pyplot as plt
from skfuzzy import gaussmf, membership

import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy
import os
import logging
import sys
import pandas as pd
import numpy as np
from skfuzzy import gaussmf

# pacotes personalizados
import util
from class_manipulate_data import ManipulateData
from class_control_panel import ControlPanel

current_path = os.path.dirname(__file__)

# region: parâmetros necessários para uso do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

file_format = logging.Formatter('%(asctime)s: %(name)s - %(levelname)s - ' +
                                '%(message)s')
log_path = os.path.join(sys.path[0], 'anfis.log')
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(file_format)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
# endregion

LENGHT_ROI = 125

manipulate_data = ManipulateData()
path_preprocessing_output = manipulate_data.get_path_preprocessing_output()

control_panel = ControlPanel(rolling_mean=False,
                             window_mean=None,
                             use_validation_data=False,
                             number_units_validation=None,
                             use_optuna=True,
                             use_savgol_filter=False,
                             use_roi=True)
 
control_panel.set_roi(LENGHT_ROI)

 

# todas as entradas
# genes = ['time',
#     'setting_1', 'setting_2', 'setting_3',
#     'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
#     'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11',
#     'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16',
#     'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']

genes = ['time', 'sensor_11']

output_model = ['RUL']

equipment_name = 'FD001'

logger.info("Lendo os dados de treino.")

path_dataset_train = \
    str(path_preprocessing_output.joinpath(f"train_{equipment_name}.parquet"))

df_train = pd.read_parquet(path_dataset_train)
df_train = control_panel.apply_roi(df_train, output_model[0], LENGHT_ROI)

logger.info("Lendo os dados de teste.")

path_dataset_test = \
str(path_preprocessing_output.joinpath(f"test_{equipment_name}.parquet"))

df_test = pd.read_parquet(path_dataset_test)
df_test = control_panel.apply_roi(df_test, output_model[0], LENGHT_ROI)

y_train = df_train[output_model].values
X_train = df_train[genes].values

y_test = df_test[output_model]
X_test = df_test[genes]


# Número de conjuntos
num_conjuntos = 5

# Criação dos conjuntos fuzzy
conjuntos = []

# Calcula a média e o desvio padrão de cada variável em X_train
medias = np.mean(X_train, axis=0)
desvios_padrao = np.std(X_train, axis=0)

# Visualização das funções de pertinência em gráficos individuais
for i in range(X_train.shape[1]):
    x_variavel = np.linspace(np.min(X_train[:, i]), np.max(X_train[:, i]), 1000)
    plt.figure(figsize=(8, 5))
    
    conjuntos_variavel = []

    # Cria 5 conjuntos Gaussianos para cada variável
    for j in range(num_conjuntos):
        media_conjunto = medias[i] + (j - (num_conjuntos - 1) / 2) * desvios_padrao[i]
        pertinencia = gaussmf(x_variavel, media_conjunto, desvios_padrao[i])
        conjuntos_variavel.append(pertinencia)

        # Impressão das funções de pertinência para cada conjunto
        # print(f'Conjunto {i + 1}, Conjunto {j + 1}:\n', pertinencia)
    
    # Adiciona os conjuntos da variável à lista
    conjuntos.append(conjuntos_variavel)

    for j in range(num_conjuntos):
        plt.plot(x_variavel, conjuntos_variavel[j], label=f'Conjunto {j + 1}')
    
    plt.title(f'Funções de Pertinência Gaussiana para Variável {genes[i]}')
    plt.xlabel('Eixo X')
    plt.ylabel('Pertinência')
    plt.legend()
    plt.grid(True)
    plt.show()