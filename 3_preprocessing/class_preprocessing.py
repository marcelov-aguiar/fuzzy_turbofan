import pandas as pd
import logging
import util
from class_format_data import FormatData

# region: parâmetros necessários para uso do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
console_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(console_format)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)
# endregion


class Preprocessing(FormatData):
    """
    Responsável por executar todas as etapas do pré-processamento
    em todos os dados.
    """
    def __init__(self,
                 ) -> None:
        """Responsável por executar todas as etapas do pré-processamento
        em todos os dados.
        """
        super().__init__()

    def run_preprocessing(self,
                          equipment_name: str):
        """
        Responsável por executar o pré-processamento para
        os dados de treino e para os dados de teste.

        Parameters
        ----------
        equipment_name : str
            Nome do equipamento que será executado o pré-processamento.
            Pode assumir os seguintes valores: FD001, FD002, FD003 ou FD004.
        """
        self.equipment_name = equipment_name
        logger.info("Iniciando processamento dos dados de treino.")
        self.training_data_preprocessing()
        logger.info("Iniciando processamento dos dados de teste.")
        self.test_data_preprocessing()
        logger.info("Processamento finalizado.")

    def training_data_preprocessing(self,
                                    remove_feat_low_variance: bool = False):
        """Responsável por executar todas etapas do processamento
        para os dados de treino.

        Parameters
        ----------
        remove_feat_low_variance : pd.DataFrame, optional.
            Caso True, remove as features de baixa variância. Caso contrário,
            as features não são removidas, by default False.
        """
        self.format_raw_data(f"train_{self.equipment_name}.txt")
        df_train = self.get_format_data(f"train_{self.equipment_name}.txt")
        if remove_feat_low_variance:
            df_train = self.remove_feat_low_variance(df_train)

        # Criação da feature RUL (Remaining  Useful  Life)
        df_train = \
            df_train.groupby('unit_number').apply(self.add_rul).reset_index()
        df_train = df_train.drop(columns=["level_1", "index"])

        # salvando dados da base de dados de treino
        path_preprocessing_output = self.get_path_preprocessing_output()

        path_aux = path_preprocessing_output.joinpath(
            f"train_{self.equipment_name}.parquet")

        df_train.to_parquet(path_aux, index=False)

    def test_data_preprocessing(self,
                                remove_feat_low_variance: bool = False):
        """Responsável por executar todas etapas do processamento
        para os dados de teste.

        Parameters
        ----------
        remove_feat_low_variance : pd.DataFrame, optional.
            Caso True, remove as features de baixa variância. Caso contrário,
            as features não são removidas, by default False.
        """
        self.format_raw_data(f"test_{self.equipment_name}.txt")
        df_test = self.get_format_data(f"test_{self.equipment_name}.txt")
        if remove_feat_low_variance:
            df_test = self.remove_feat_low_variance(df_test)

        # Criação da feature RUL (Remaining  Useful  Life)
        df_test_rul = pd.DataFrame(
            df_test.groupby('unit_number')['time'].max()).reset_index()
        df_test_rul.columns = ['id', 'max']

        df_rul = self.read_txt_rul(f'RUL_{self.equipment_name}.txt')
        df_rul['TOTAL_RUL'] = df_test_rul['max'] + df_rul['max']
        df_rul = df_rul.drop(columns=["max"])

        df_test.merge(df_rul, on=['unit_number'], how='left')
        df_test = df_test.merge(df_rul, on=['unit_number'], how='left')
        df_test['RUL'] = df_test['TOTAL_RUL'] - df_test['time']
        df_test.drop('TOTAL_RUL', axis=1, inplace=True)

        # salvando dados da base de dados de teste
        path_preprocessing_output = self.get_path_preprocessing_output()

        path_aux = path_preprocessing_output.joinpath(
            f"test_{self.equipment_name}.parquet")

        df_test.to_parquet(path_aux, index=False)

    def remove_feat_low_variance(self,
                                 df_data: pd.DataFrame,
                                 VAR: float = 0.00000001) -> pd.DataFrame:
        """Responsável por remover as features de `df_data` com
        baixa variância.

        Parameters
        ----------
        df_data : pd.DataFrame
            DataFrame com as features a serem removidas.
        VAR : float, optional
            Valor da variância considerado como baixo. Abaixo desse valor
            as variáveis (features) serão removidas, by default 0.00000001.

        Returns
        -------
        pd.DataFrame
            DataFrame sem as features com baixa variância.
        """
        good_sensor = list(df_data.columns)
        for sensor in df_data.columns:
            if df_data[sensor].var() <= VAR:
                logger.info(f"Sensor is flat: {sensor}")
                good_sensor.remove(sensor)
        df_data = df_data[good_sensor]
        return df_data

    def add_rul(self,
                df_unit: pd.DataFrame) -> pd.DataFrame:
        """Responsável por criar a feature RUL (Remaining
        Useful Life) para cada unidade do DataFrame principal.

        Parameters
        ----------
        df_unit : pd.DataFrame
            DataFrame de uma unidade (parte do DataFrame principal).

        Returns
        -------
        pd.DataFrame
            DataFrame da unidade com RUL calculado.
        """
        df_unit['RUL'] = [max(df_unit['time'])] * len(df_unit)
        df_unit['RUL'] = df_unit['RUL'] - df_unit['time']
        del df_unit['unit_number']
        return df_unit.reset_index()


if __name__ == '__main__':
    preprocessing = Preprocessing()
    preprocessing.run_preprocessing('FD001')
