from operator import attrgetter
import random
from deap import algorithms, base, creator, tools
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from deap import base, creator, tools, algorithms
import logging
import random
import numpy as np
import sys
from typing import List, Tuple
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
import sys
import os 

# pacotes personalizados
import util
from class_manipulate_data import ManipulateData
from class_control_panel import ControlPanel

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
log_path = os.path.join(sys.path[0], 'processing.log')
file_handler = logging.FileHandler(log_path)
file_handler.setFormatter(file_format)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
# endregion

logger.info(util.init())


class Chromosome(object):
    """Implements a chromosome container.

    Chromosome represents the list of genes, whereas each gene is
    the name of feature. Creating the chromosome, we generate
    the random sample of features.

    Args:
        genes:
            List of all feature names.
        size:
            Number of genes in chromosome,
            i.e. number of features in the model.
    """

    def __init__(self, genes, size):
        self.genes = self.generate(genes, size)

    def __repr__(self):
        return ' '.join(self.genes)

    def __get__(self, instance, owner):
        return self.genes

    def __set__(self, instance, value):
        self.genes = value

    def __getitem__(self, item):
        return self.genes[item]

    def __setitem__(self, key, value):
        self.genes[key] = value

    def __len__(self):
        return len(self.genes)

    @staticmethod
    def generate(genes, size):
        return random.sample(genes, size)


def init_individual(ind_class, genes=None, size=None):
    return ind_class(genes, size)


def evaluate(individual,
             model,
             X_train: pd.DataFrame,
             y_train: pd.DataFrame,
             X_test: pd.DataFrame,
             y_test: pd.DataFrame) -> Tuple[float, None]:
    
    X_train = X_train[individual.genes]
    model.fit(X_train, y_train)

    X_test = X_test[individual.genes]
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=True)
    return rmse,


def mutate(individual, genes=None, pb=0):
    """Custom mutation operator used instead of standard tools.

    We define the maximal number of genes, which can be mutated,
    then generate a random number of mutated genes (from 1 to max),
    and implement a mutation.

    Args:
        individual:
            The list of features (genes).
        genes:
            The list of all features.
        pb:
            Mutation parameter, 0 < pb < 1.

    Returns:
         Mutated individual (tuple).
    """

    # set the maximal amount of mutated genes
    n_mutated_max = max(1, int(len(individual) * pb))

    # generate the random amount of mutated genes
    n_mutated = random.randint(1, n_mutated_max)

    # pick up random genes which need to be mutated
    mutated_indexes = random.sample(
        [index for index in range(len(individual.genes))], n_mutated)

    # mutation
    for index in mutated_indexes:
        individual[index] = random.choice(genes)

    return individual,


def select_best(individuals, k, fit_attr='fitness'):
    """Custom selection operator.

    The only difference with standard 'selBest' method
    (select k best individuals) is that this method doesn't select
    two individuals with equal fitness value.

    It is done to prevent populations with many duplicate individuals.
    """

    return sorted(
        set(individuals),
        key=attrgetter(fit_attr),
        reverse=True)[:k]


def main():

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

    n_features = 7

    # todas as entradas
    genes = ['time',
        'setting_1', 'setting_2', 'setting_3',
        'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5', 'sensor_6',
        'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10', 'sensor_11',
        'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_16',
        'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21']

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

    y_train = df_train[output_model]
    X_train = df_train[genes]

    y_test = df_test[output_model]
    X_test = df_test[genes]

    model = RandomForestRegressor(max_depth=8,
                                  n_estimators=186,
                                  max_features=5,
                                  min_samples_split=6)
    pipeline = Pipeline([('std', StandardScaler()), ('regressor', model)])

    pipeline = TransformedTargetRegressor(regressor=pipeline,
                                          transformer=StandardScaler())
    model = pipeline

    # setting individual creator
    creator.create('FitnessMin', base.Fitness, weights=(-1,))
    creator.create('Individual', Chromosome, fitness=creator.FitnessMin)

    # register callbacks
    toolbox = base.Toolbox()
    toolbox.register(
        'individual', init_individual, creator.Individual,
        genes=genes, size=n_features)
    toolbox.register(
        'population', tools.initRepeat, list, toolbox.individual)

    # raise population
    pop = toolbox.population(50)

    # keep track of the best individuals
    hof = tools.HallOfFame(5)

    # register fitness evaluator
    toolbox.register("evaluate",
                     evaluate,
                     model=model,
                     X_train=X_train,
                     y_train=y_train,
                     X_test=X_test,
                     y_test=y_test)
    # register standard crossover
    toolbox.register('mate', tools.cxTwoPoint)
    # replace mutation operator by our custom method
    toolbox.register('mutate', mutate, genes=genes, pb=0.4)
    # register elitism operator
    toolbox.register('select', select_best)

    # setting the statistics (displayed for each generation)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register('avg', np.mean, axis=0)
    stats.register('min', np.min, axis=0)
    stats.register('max', np.max, axis=0)

    # mu: the number of individuals to select for the next generation
    # lambda: the number of children to produce at each generation
    # cxpb: the probability that offspring is produced by crossover
    # mutpb: the probability that offspring is produced by mutation
    # ngen: the number of generations
    #try:
    algorithms.eaMuPlusLambda(
        pop, toolbox,
        mu=10, lambda_=30, cxpb=0.2, mutpb=0.8,
        ngen=150, stats=stats, halloffame=hof, verbose=True)
    # except (Exception, KeyboardInterrupt):
    #     for individual in hof:
    #         logger.info(
    #             f'hof: {individual.fitness.values[0]:.3f} << {individual}')
    #     print("Erro")
    for individual in hof:
        logger.info(
            f'hof: {individual.fitness.values[0]:.3f} << {individual}')
    """ 
    Features selecionadas:
    sensor_12: Razão da vazão do combustível para Ps30 (pps/psia)
    time: tempo de operação do motor (ciclos)
    sensor_7: Pressão de saída do HPC (psia)
    sensor_2: Temperatura de saída do LPC (°R)
    sensor_11: Pressão estática do HPC (psia)
    sensor_4: Temperatura de saída da LPT (°R)
    sensor_9: Velocidade física do núcle (rpm)
    """

if __name__ == '__main__':
    main()