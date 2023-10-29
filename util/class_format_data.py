import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import logging
import util
from class_manipulate_data import ManipulateData

# region: parâmetros necessários para uso do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
# endregion


class FormatData(ManipulateData):
    """É reponsável por formatar os dados brutos para o formato
    de DataFrame para ser possível fazer a análise exploratória.

    Parameters
    ----------
    ManipulateData : object
        Guardar informações da localização dos
        repositórios e dos dados específicos do projeto.
    """
    def __init__(self):
        super().__init__()

    def read_txt_rul(self,
                     input_file_name: Path) -> pd.DataFrame:
        """Responsável por fazer a leitura do arquivo RUL dos
        dados de teste.

        Parameters
        ----------
        input_file_name : Path
            Nome do arquivo que contém os dados do equipamento que
            será lido os dados. Pode assumir os seguintes valores:
            RUL_FD00Y.txt. Onde Y pode assumir os valores 1, 2, 3 ou 4.

        Returns
        -------
        pd.DataFrame
            DataFrame com os dados brutos.
        """
        path_raw_data = self.get_path_raw_data()

        path_data = path_raw_data.joinpath(input_file_name)

        data = np.loadtxt(path_data)

        df_data = pd.DataFrame(data, columns=['max'])

        features_name = self.get_features_name()
        df_data[features_name[0]] = df_data.index.values + 1

        return df_data

    def format_raw_data(self,
                        input_file_name: str) -> None:
        """Responsável por fazer a leitura do dado bruto no formato
        .txt, renomear as colunas para os nomes definidos em
        `get_features_name` e salvar o arquivo em formato csv.

        Parameters
        ----------
        input_file_name : Path
            Nome do arquivo que contém os dados do equipamento que
            será lido os dados. Pode assumir os seguintes valores:
            X_FD00Y.txt. Onde X pode assumir valor `test` ou `train`.
            E Y pode assumir os valores 1, 2, 3 ou 4.
        """
        path_raw_data = self.get_path_raw_data()

        path_data = path_raw_data.joinpath(input_file_name)

        data = np.loadtxt(path_data)

        features_name = self.get_features_name()

        df_data = pd.DataFrame(data, columns=features_name)

        get_path_exploratory_data = self.get_path_exploratory_data()

        input_file_name = os.path.splitext(input_file_name)[0]
        output_file_name = f"{input_file_name}_format.parquet"

        path_output_format_data = \
            get_path_exploratory_data.joinpath(output_file_name)

        df_data.to_parquet(path_output_format_data, index=False)

    def get_format_data(self,
                        output_file_name: str) -> pd.DataFrame:
        """Responsável por fazer a leitura do arquivo salvo por
        `format_raw_data`.

        Parameters
        ----------
        output_file_name : str
            Nome do arquivo do equipamento que será feito a leitura.

        Returns
        -------
        pd.DataFrame
            DataFrame dos dados do equipamento.
        """
        get_path_exploratory_data = self.get_path_exploratory_data()

        output_file_name = os.path.splitext(output_file_name)[0]
        try:
            df_data = pd.read_parquet(
                os.path.join(get_path_exploratory_data,
                             f"{output_file_name}_format.parquet"))

        except Exception as e:
            logger.error("Dado não encontrado. Execute método " +
                         f"format_raw_data para {output_file_name}.")
            logger.error(e)

        return df_data
