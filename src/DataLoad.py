import pandas as pd
import numpy as np
import os
import kagglehub

class DataLoader:
    # used for loading, validating and cleaning the data

    def __init__(self):
        self._df: pd.DataFrame | None = None

    @property
    def dataframe(self):
        ''' raises RuntimeError if load() hasn't been called'''
        if self._df is None:
            raise RuntimeError("Data not loaded yet. Call load() first.")
        return self._df

    @property
    def shape(self):
        ''' Returns rows x columns shape of the DataFrame'''
        return self.dataframe.shape

    def __repr__(self):
        loaded = f"loaded {self._df.shape}" if self._df is not None else "not loaded"
        return f"DataLoader(kaggle, data={loaded})"

    def __len__(self):
        return len(self.dataframe)


    
    def load(self):
        ''' loads the file from Kaggle and strips whitespace from the column names.
        Returns:
            pd.DataFrame: the loaded DataFrame
        '''
        
        path = kagglehub.dataset_download("fida5073/female-infertility")
        csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
        self._df = pd.read_csv(os.path.join(path, csv_file))
        self._df.columns = self._df.columns.str.strip()
        print(f"Loaded: {self._df.shape[0]} rows × {self._df.shape[1]} columns")
        return self._df

    def clean(self):
        ''' drops a column with patient ID and columns with many missing values.
        Returns:
            pd.DataFrame: the cleaned DataFrame
        '''
        df = self.dataframe
        self._df.drop(columns=['Patient ID'], inplace=True, errors='ignore')

        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        before = len(df)
        dropped = before - len(self._df)
        if dropped:
            print(f"Removed {dropped} rows")
        return self._df

    def split_features_target(self, target_column: str):
        ''' splits the DataFrame into feature matrix (X) and target vector (Y).
        Args:
        target_column (str): name of the column to use as a target variable

        Returns:
            tuple[pd.DataFrame, pd.Series]: A tuple (X, y) where X is all
            columns except the target and y contains the target column values
        '''
        
        df = self.dataframe
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found. Columns in the dataset: {list(df.columns)}")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
