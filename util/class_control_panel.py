import pandas as pd
from scipy.signal import savgol_filter
from typing import List, Tuple

class ControlPanel():
    """É responsável por definir parâmetros para fazer diferentes
    experimentos
    """
    def __init__(self,
                 rolling_mean: bool = False,
                 window_mean: int = 0,
                 is_grid_search: bool = False,
                 use_validation_data: bool = False,
                 use_optuna: bool = False,
                 number_units_validation: int = 0,
                 use_savgol_filter: bool = False,
                 use_roi: bool = False) -> None:
        """É responsável por definir parâmetros para fazer diferentes
        experimentos.

        Parameters
        ----------
        rolling_mean : bool, optional.
            Caso True, será feita a média móvel dos dados conforme o número da
            unidade, by deafult False.
        window_mean : int, optional, optional.
            Tamanho da janela que será feita a média móvel, by default 0
        is_grid_search : bool, optional.
            Caso True, será feito GridSearch com validação cruzada, by default False.
        use_validation_data : bool, optional.
            Caso True, parte dos dados de treino é movido para os dados de teste,
            by default False
        use_optuna : bool, optional.
            Caso True, é usado o Optuna para otimizar os hiperparâmetros,
            by default False
        number_units_validation : int, optional
            Número de unidades que será migrado dos dados de treino para os
            dados de teste.
        apply_use_savgol_filter: bool, optional
            Caso True, filtro Savitzky–Golay é aplicado para diminuir o ruído dos
            dados, by default False
        use_roi: bool, optional
            Caso True, somente os dados da região de interesse (ROI) são
            retornados do DataFrame, ou seja, é feito um filtro para RUL
            menor de determinado valor (por padrão é 150).
        """
        self.rolling_mean = rolling_mean
        self.window_mean = window_mean
        self.is_grid_search = is_grid_search
        self.use_validation_data = use_validation_data
        self.use_optuna = use_optuna
        self.number_units_validation = number_units_validation
        self.use_savgol_filter = use_savgol_filter
        self.use_roi = use_roi

    def apply_use_savgol_filter(self,
                                df_data: pd.DataFrame,
                                ignore_column: str,
                                window_length: int = 12,
                                polyorder: int = 2) -> pd.DataFrame:
        
        if self.use_savgol_filter:
            new_columns = []
            for column in df_data.columns:
                if not (column == ignore_column):
                    df_data[f'{column}_savgol'] = savgol_filter(df_data[column],
                                                                window_length,
                                                                polyorder)
                    new_columns.append(f'{column}_savgol')
                else:
                    new_columns.append(ignore_column)
            return df_data[new_columns]
        return df_data

    def apply_use_validation_data(self,
                                  df_train: pd.DataFrame,
                                  df_test: pd.DataFrame,
                                  column_name: str = 'unit_number') \
                                    -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.use_validation_data:
            units_quantity = self.number_units_validation
            units_numbers = df_train[column_name].unique()[-units_quantity:]
            for unit_number in units_numbers:
                df_aux = df_train[df_train[column_name] == unit_number].copy()
                df_train = df_train[~(df_train[column_name] == unit_number)]
                df_aux[column_name] = df_aux[column_name] + 100
                df_test = pd.concat([df_test, df_aux], axis=0)
        
        return df_train, df_test
    
    def apply_rolling_mean(self,
                           df_data: pd.DataFrame,
                           name_column: str) -> pd.DataFrame:
        """É reponsável por aplicar a média móvel nos
        dados.

        Parameters
        ----------
        df_data : pd.DataFrame
            DataFrame com os dados a serem realizados a
            média móvel.
        name_column : str
            Nome da coluna que representa o número da
            unidade para fazer o agrupamento.

        Returns
        -------
        pd.DataFrame
            DataFrame aplicado com a média móvel
        """
        if self.rolling_mean:
            df_rolling = \
                df_data.groupby(name_column).rolling(
                    window=self.window_mean).mean()

            df_rolling = df_rolling.dropna()
            df_rolling = df_rolling.reset_index()
            df_data = df_rolling.copy()
        return df_data
    
    def apply_roi(self,
                  df_data: pd.DataFrame,
                  rul_column_name: str,
                  value_rul: int) -> pd.DataFrame:
        """Responsável por filtrar os dados somente na
        região de interesse, ou seja, considera o RUL abaixo
        de `value_rul`.

        Parameters
        ----------
        df_data : pd.DataFrame
            _description_
        rul_column_name : str
            _description_
        value_rul : int
            _description_

        Returns
        -------
        pd.DataFrame
            _description_
        """
        if self.use_roi:
            df_data[rul_column_name][df_data[rul_column_name] > value_rul] = value_rul
        return df_data

    def set_roi(self, value: int) -> None:
        self.LENGHT_ROI: int = value
    
    def get_roi(self) -> int:
        return self.LENGHT_ROI