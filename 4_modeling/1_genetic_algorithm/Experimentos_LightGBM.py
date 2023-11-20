# %% [markdown]
# Repositório: https://github.com/mdrs-thiago/PUC_FuzzyLogic

# %%
import pandas as pd
import logging
import random
import numpy as np
import sys
from typing import List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import os

# pacotes personalizados
import util
from class_manipulate_data import ManipulateData
from class_control_panel import ControlPanel
from ga_K_LightGBM import GA
# %%
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
log_path = os.path.join(sys.path[0], 'ga_LightGBM.log')
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(file_format)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
# endregion

logger.info(util.init())

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

# %%
# tamanho da região de interesse (RUL abaixo de LENGHT_ROI)
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

for k in range(2,20):
    input_model = GA(k)
    # %%
    # entradas selecionadas pelo algoritmo genético

    output_model = ['RUL']

    equipment_name = 'FD001'
   
    # %%
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

    # %%
    df_train = df_train[input_model + output_model].copy()

    df_test = df_test[input_model + output_model].copy()

    model = LGBMRegressor(max_depth=4, n_estimators=36, boosting_type="gbdt")

    pipeline = Pipeline([('std', StandardScaler()), ('regressor', model)])

    pipeline = TransformedTargetRegressor(regressor=pipeline,
                                          transformer=StandardScaler())
    model = pipeline

    y_train = df_train[output_model]
    X_train = df_train[input_model]

    y_test = df_test[output_model]
    X_test = df_test[input_model]

    model.fit(X_train, y_train)
    # %%
    y_train_pred = model.predict(X_train)
    
    if control_panel.use_roi:
        X_train['REAL'] = y_train.values
        X_train['PREDITO'] = y_train_pred
        X_train = X_train[~(X_train['REAL'] == control_panel.LENGHT_ROI)]
        y_train_select = X_train['REAL']
        y_train_pred = X_train['PREDITO'].values
        X_train = X_train.drop(columns=['REAL', 'PREDITO'])

    rmse = root_mean_squared_error(y_train_select, y_train_pred)
    mae = mean_absolute_error(y_train_select, y_train_pred)
    logger.info(f"RMSE train: {rmse}")
    logger.info(f"MAE train: {mae}")

    # %%
    y_test_pred = model.predict(X_test)

    if control_panel.use_roi:
        X_test['REAL'] = y_test.values
        X_test['PREDITO'] = y_test_pred
        X_test = X_test[~(X_test['REAL'] == control_panel.LENGHT_ROI)]
        y_test_select = X_test['REAL']
        y_test_pred = X_test['PREDITO'].values
        X_test = X_test.drop(columns=['REAL', 'PREDITO'])

    rmse = root_mean_squared_error(y_test_select, y_test_pred)
    mae = mean_absolute_error(y_test_select, y_test_pred)
    logger.info(f"RMSE teste: {rmse}")
    logger.info(f"MAE teste: {mae}")


