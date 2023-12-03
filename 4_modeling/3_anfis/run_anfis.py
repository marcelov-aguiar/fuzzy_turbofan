import anfis
import membership.mfDerivs
import membership.membershipfunction
import numpy
import os
import logging
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error

import warnings

warnings.filterwarnings("ignore")

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

def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Faz o cálculo do RMSE.

    Parameters
    ----------
    y_true : pd.Series
        Valor real.
    y_pred : pd.Series
        Valor predito pelo modelo.

    Returns
    -------
    float
        Valor do RMSE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

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

#genes = ['time', 'sensor_11']
genes = ['sensor_9', 'sensor_11', 'time']
#genes = ['sensor_9', 'sensor_11', 'sensor_12', 'time']
#genes = ['sensor_7', 'sensor_11', 'sensor_9', 'time', 'sensor_12']
output_model = ['RUL']

equipment_name = 'FD002'

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

y_test = df_test[output_model].values
X_test = df_test[genes]

# mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
#             [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]
# 
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
    
    conjuntos_variavel = []

    # Cria 5 conjuntos Gaussianos para cada variável
    for j in range(num_conjuntos):
        media_conjunto = medias[i] + (j - (num_conjuntos - 1) / 2) * desvios_padrao[i]
        #pertinencia = gaussmf(x_variavel, media_conjunto, desvios_padrao[i])
        pertinencia = ['gaussmf',{'mean':media_conjunto,'sigma':desvios_padrao[i]}]
        conjuntos_variavel.append(pertinencia)

        # Impressão das funções de pertinência para cada conjunto
        # print(f'Conjunto {i + 1}, Conjunto {j + 1}:\n', pertinencia)
    
    # Adiciona os conjuntos da variável à lista
    conjuntos.append(conjuntos_variavel)

mf = conjuntos.copy()

mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X_train, np.ravel(y_train), mfc)
anf.trainHybridJangOffLine(epochs=2)
# print(round(anf.consequents[-1][0],6))
# print(round(anf.consequents[-2][0],6))
# print(round(anf.fittedValues[9][0],6))
if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
	print('test is good')

X_train_select = pd.DataFrame(X_train, columns=genes)

pred_rul_train = anfis.predict(anf, X_train)

if control_panel.use_roi:
    X_train_select['REAL'] = y_train
    X_train_select['PREDITO'] = pred_rul_train
    X_train_select = X_train_select[~(X_train_select['REAL'] == control_panel.LENGHT_ROI)]
    y_train_select = X_train_select['REAL']
    y_train_pred = X_train_select['PREDITO'].values
    X_train_select = X_train_select.drop(columns=['REAL', 'PREDITO'])

rmse = root_mean_squared_error(y_train_select, y_train_pred)
mae = mean_absolute_error(y_train_select, y_train_pred)
print("MAE Train", mae)
print("RMSE Train", rmse)

y_test_pred = anfis.predict(anf, X_test.values)

if control_panel.use_roi:
        X_test['REAL'] = y_test
        X_test['PREDITO'] = y_test_pred
        X_test = X_test[~(X_test['REAL'] == control_panel.LENGHT_ROI)]
        y_test = X_test['REAL']
        y_test_pred = X_test['PREDITO'].values
        X_test = X_test.drop(columns=['REAL', 'PREDITO'])

mae = mean_absolute_error(y_test, y_test_pred)
rmse = root_mean_squared_error(y_test, y_test_pred)
print("MAE Test", mae)
print("RMSE Test", rmse)
