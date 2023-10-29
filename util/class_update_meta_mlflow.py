import os
import yaml
from typing import Dict, Any, List
from yaml.loader import SafeLoader
import re
import glob
from itertools import chain
import logging


# region: parâmetros necessários para uso do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
# endregion


class UpdateMetaMLFlow():
    """Responsável por atualizar o path dos arquivos meta.yaml
    do MLFlow.
    """
    def __init__(self) -> None:
        pass

    def process_update(self, path_mlflow: str):
        """Responsável por executar o processamento que atualiza
        o path dos arquivos meta.yaml para pessoa que está executando
        o MLFlow.

        Parameters
        ----------
        path_mlflow : str
            Path onde está a pasta mlruns do MLFlow.
        """
        files_meta_path = self.search_file_meta_yaml(path_mlflow)
        for file_path in files_meta_path:
            data = self.read_yaml(file_path)
            data = self.replace_uri_meta(data, path_mlflow)
            self.write_yaml(file_path, data)
        logger.info("Atualização do arquivo meta.yaml realizada com sucesso.")

    def replace_uri_meta(self, file_yaml: Dict[str, Any], path_mlflow: str) -> Dict[str, Any]:
        """Responsável por substituir o path dos arquivos meta.yaml para o path
        da máquina do usuário atual do MLFlow.

        Parameters
        ----------
        file_yaml : Dict[str, Any]
            Dado do meta.yaml no formato de dicionário.
        path_mlflow : str
            Path onde está a pasta mlruns do MLFlow.

        Returns
        -------
        Dict[str, Any]
            Dado do meta.yaml no formato de dicionário com o path atualizado
            para máquina da pessoa que está executando o MLFlow.
        """
        keys = ['artifact_uri', 'artifact_location']

        for key in keys:
            if key in file_yaml:
                string = file_yaml[key]
                match = (re.search("mlruns", string))
                end_mlruns = match.span()[1]
                path_mlflow_full = 'file:///' + path_mlflow
                file_yaml[key] = \
                    string.replace(string[:end_mlruns], path_mlflow_full)
                return file_yaml
        raise("Nenhuma chave foi encontrada no arquivo yaml.")


    def write_yaml(self, path_yaml: str, data: dict):
        """Escreve o conteúdo de `data` em um arquivo yaml no
        path `path_yaml`.

        Parameters
        ----------
        path_yaml : str
            Path onde será salvo o arquivo yaml.
        data : dict
            Dado a ser salvo no arquivo yaml.
        """
        with open(path_yaml, 'w') as f:
            data = yaml.dump(data,
                             f,
                             sort_keys=False,
                             default_flow_style=False)


    def read_yaml(self, path: str) -> Dict[str, Any]:
        """Lê um arquivo yaml dado o seu caminho (path)

        Returns
        -------
        Dict[str, Any]
            Retorna o arquivo yaml em um dicionário Python.
        """
        with open(path) as f:
            data = yaml.load(f, Loader=SafeLoader)

        return data


    def search_file_meta_yaml(self, path_mlflow: str) -> List[str]:
        """Responsável por pesquisar arquivos no formato `meta.yaml`
        no diretório `mlruns` do MLFlow.

        Parameters
        ----------
        path_mlflow : str
            Diretório do `mlruns` onde se localiza os experimentos.

        Returns
        -------
        List[str]
            Lista com a localização de todos arquivos `meta.yaml`
            dentro do diretório `path_mlflow`.
        """
        path_mlflow = os.path.join(path_mlflow, '**')
        files = [f for f in [glob.glob(os.path.join(
                    path_mlflow, 'meta.yaml'), recursive=True)]]
        files = sorted(chain(*files))
        return files