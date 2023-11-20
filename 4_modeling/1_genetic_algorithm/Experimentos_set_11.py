# %% [markdown]
# Repositório: https://github.com/mdrs-thiago/PUC_FuzzyLogic

# %%
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import math
import pandas as pd
from deap import base, creator, tools, algorithms
import logging
import random
import numpy as np
import sys
from typing import List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import RidgeCV
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
log_path = os.path.join(sys.path[0], 'ga_v3.log')
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(file_format)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
# endregion

logger.info(util.init())

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

for k in range(7,20):
    input_model = GA(k)
    # %%
    # entradas selecionadas pelo algoritmo genético

    output_model = ['RUL']

    equipment_name = 'FD001'

    
    #set_amounts = 5
    set_amounts = 11
    # 
    #set_names = ['Baixo','Moderado', 'OK','Elevado', 'Alto']
    set_names = ["Teste1", "Extremamente baixo",'Muito baixo','Baixo','Moderado', 'OK','Elevado', 'Alto', 'Muito alto', "Extremamente alto", "Teste2"]
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

    # %%
    var_linguisticas = []
    for input in input_model:
        min_train = df_train[input].min()
        min_test = df_test[input].min()
        if min_test < min_train:
            min_r = min_test
        else:
            min_r = min_train

        max_train = df_train[input].max()
        max_test = df_test[input].max()
        if max_test < max_train:
            max_r = max_train
        else:
            max_r = max_test
        aux = {}
        aux["name"] = input
        aux["min"] = math.floor(min_r) - 1
        aux["max"] = math.ceil(max_r) + 1
        aux["step"] = 0.01
        var_linguisticas.append(aux)

    # %%
    # Definimos o universo de discurso e o nome das variáveis linguísticas.
    for var in var_linguisticas:
        var["antecedente"] = ctrl.Antecedent(np.arange(var["min"], var["max"], var["step"]), var["name"])
        var["antecedente"].automf(set_amounts, names=set_names)

    RUL = ctrl.Consequent(np.arange(math.floor(df_train[output_model].values.min()),
                                    math.ceil(df_train[output_model].values.max()) + 1,
                                    0.01),
                                    output_model[0],
                                    defuzzify_method='centroid')
    RUL.automf(set_amounts, names=set_names)

    # %% [markdown]
    # # Extração de regras

    # %%
    def manter_regras_com_maior_DR_j(regras):
        antecedentes_para_regras = {}

        # Passo 1: Criar um dicionário mapeando antecedentes para as regras com consequentes diferentes
        for regra in regras:
            antecedentes = tuple(regra['antecedentes'])
            consequente = regra['consequente']
            if antecedentes not in antecedentes_para_regras:
                antecedentes_para_regras[antecedentes] = {'regras': [regra], 'maior_DR_j': regra['DR_j']}
            else:
                antecedentes_para_regras[antecedentes]['regras'].append(regra)
                if regra['DR_j'] > antecedentes_para_regras[antecedentes]['maior_DR_j']:
                    antecedentes_para_regras[antecedentes]['maior_DR_j'] = regra['DR_j']

        # Passo 2: Criar uma nova lista de regras com os maiores DR_j
        novas_regras = []
        for antecedentes, info in antecedentes_para_regras.items():
            maior_DR_j = info['maior_DR_j']
            regras_com_maior_DR_j = [regra for regra in info['regras'] if regra['DR_j'] == maior_DR_j]
            novas_regras.extend(regras_com_maior_DR_j)

        return novas_regras

    # %%
    logger.info("Extraindo as regras")
    regras = []
    for indice, data in df_train.iterrows():
        DR_j = 1
        regra = {}
        # antecedente
        regra["antecedentes"] = []
        for var in var_linguisticas:
            valor = data[var["name"]]
            grau_max = 0
            conj_max = None
            for conjunto in var["antecedente"].terms:
                grau_pertinencia = fuzz.interp_membership(var["antecedente"].universe, var["antecedente"][conjunto].mf, valor)
                if grau_pertinencia > grau_max:
                    grau_max = grau_pertinencia
                    conj_max = conjunto
            DR_j = DR_j * grau_max

            regra["antecedentes"].append(var["antecedente"][conj_max])

        # consequente
        valor = data["RUL"]
        grau_max = 0
        conj_max = None
        for conjunto in RUL.terms:
            grau_pertinencia = fuzz.interp_membership(RUL.universe, RUL[conjunto].mf, valor)
            if grau_pertinencia > grau_max:
                grau_max = grau_pertinencia
                conj_max = conjunto
        DR_j = DR_j * grau_max
        regra["DR_j"] = DR_j
        regra["consequente"] = RUL[conj_max]
        regras.append(regra)

    # %%
    logger.info("Filtrando as regras")
    novas_regras = manter_regras_com_maior_DR_j(regras)

    # %%
    logger.info(f"São {len(novas_regras)} regras após a filtragem.")

    # %%
    ctrl_rule = []
    # Combine os antecedentes usando o operador E ("&")
    for regra in novas_regras:
        antecedentes = regra["antecedentes"]
        consequente = regra["consequente"]
        regra_aux = antecedentes[0]
        for antecedente in antecedentes[1:]:
            regra_aux = regra_aux & antecedente

        # Combine o antecedente com o consequente
        # regra_final = regra >> consequente

        # Adicione a regra ao controlador (se necessário)
        ctrl_rule.append(ctrl.Rule(regra_aux, consequente))

    # %%
    len(ctrl_rule)

    # %% [markdown]
    # # System

    # %%
    # Criando o sistema de controle
    logger.info("Criando o sistema de controle")
    sistema_controle = ctrl.ControlSystem(ctrl_rule)
    controle = ctrl.ControlSystemSimulation(sistema_controle)

    # %%
    data = df_train.iloc[0]

    # %%
    logger.info("Calculando previsão nos dados de treino.")
    pred_rul = []
    for indice, data in df_train.iterrows():
        DR_j = 1
        regra = {}
        # antecedente
        regra["antecedentes"] = []
        for var in var_linguisticas:
            valor = data[var["name"]]
            controle.input[var["name"]] = valor
        controle.compute()
        # Obtendo a saída
        pred_rul.append(controle.output['RUL'])

    # %%
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
    X_train_select = df_train[input_model]
    y_train = df_train["RUL"]

    if control_panel.use_roi:
        X_train_select['REAL'] = y_train.values
        X_train_select['PREDITO'] = pred_rul
        X_train_select = X_train_select[~(X_train_select['REAL'] == control_panel.LENGHT_ROI)]
        y_train_select = X_train_select['REAL']
        y_train_pred = X_train_select['PREDITO'].values
        X_train_select = X_train_select.drop(columns=['REAL', 'PREDITO'])

    rmse = root_mean_squared_error(y_train_select, y_train_pred)
    mae = mean_absolute_error(y_train_select, y_train_pred)
    logger.info(f"RMSE {rmse}")
    logger.info(f"MAE {mae}")

    # %%
    logger.info("Calculando previsão nos dados de teste.")
    pred_rul = []
    for indice, data in df_test.iterrows():
        DR_j = 1
        regra = {}
        # antecedente
        regra["antecedentes"] = []
        for var in var_linguisticas:
            valor = data[var["name"]]
            controle.input[var["name"]] = valor
        controle.compute()
        # Obtendo a saída
        pred_rul.append(controle.output['RUL'])

    # %%
    X_test_select = df_test[input_model]
    y_test = df_test["RUL"]

    if control_panel.use_roi:
        X_test_select['REAL'] = y_test.values
        X_test_select['PREDITO'] = pred_rul
        X_test_select = X_test_select[~(X_test_select['REAL'] == control_panel.LENGHT_ROI)]
        y_test_select = X_test_select['REAL']
        y_test_pred = X_test_select['PREDITO'].values
        X_test_select = X_test_select.drop(columns=['REAL', 'PREDITO'])

    rmse = root_mean_squared_error(y_test_select, y_test_pred)
    mae = mean_absolute_error(y_test_select, y_test_pred)
    logger.info(f"RMSE {rmse}")
    logger.info(f"MAE {mae}")


