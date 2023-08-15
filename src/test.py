from config import paths
from utils import read_csv_in_directory, save_dataframe_as_csv
from Classifier import Classifier
from schema.data_schema import load_saved_schema
from utils import set_seeds
from sklearn.preprocessing import MinMaxScaler
from joblib import load
import pandas as pd
from pycaret.classification import compare_models, setup



data = pd.read_csv('/Users/moo/Desktop/data/nba/nba_train.csv')

setup(data, target='TARGET_5Yrs')

best_model = compare_models()

