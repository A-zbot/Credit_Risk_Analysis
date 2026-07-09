import numpy as np
import pandas as pd

from config import DATASET_PATH, TARGET_COLUMN

def load_data():
    try:
        data = pd.read_csv(DATASET_PATH, skipinitialspace=True)
        print ("=" * 60)
        print ("Data loaded successfully!")
        print (f"row count: {data.shape[0]}")
        print (f"column count: {data.shape[1]}")
        print ("=" * 60)

        return data
    except FileNotFoundError:   
        raise FileNotFoundError(f"File not found at {DATASET_PATH}. Please check the path and try again.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file at {DATASET_PATH} is empty. Please provide a valid dataset.")
    except Exception as error:
        raise Exception(f"An error occurred while loading the data: {error}")         
        

def data_info(data):
    print("\nFirst Five Records:")
    print("=" * 60)
    print(data.head())
    print("\nData Information:")
    print("=" *60)
    print(data.info())
    print("\nCounting Missing Values:")
    print("=" * 60)
    print(data.isnull().sum())
    print("\nStatistical Summary:")
    print("=" * 60)
    print(data.describe(include = "all"))

def get_feature_target(data, target_column):
        X = data.drop(columns=[target_column])
        y = data[target_column]
        return X, y
'''
machine learning needs two things:
1. Features (X): These are the input variables that the model will use to make predictions. They can be numerical or categorical data that represent the characteristics of the data points.
2. Target (y): This is the variable that the model will try to predict based on the features. It is usually a numerical value that represents the outcome or response variable.
'''
if __name__ == "__main__":
    df = load_data()
    data_info(df)
    X, y = get_feature_target(df, TARGET_COLUMN)
    print("\nfeatures shape:", X.shape)
    print("target shape:", y.shape)
