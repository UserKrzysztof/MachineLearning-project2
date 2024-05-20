import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import re

class StringiEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, top: int, delimiter: str = ',') -> None:
        super().__init__()
        self.top = top
        self.column = column
        self.delimiter = delimiter
        self.cols = []

    def fit(self,X, y=None):
        return self
    
    def transform(self, X):
        code_column = pd.DataFrame(X[self.column].str.split(self.delimiter).tolist()).stack()
        code_column = code_column.value_counts().head(self.top).index.tolist()

        for code in code_column:
            X[f'is_{self.column}_{code}'] = X[self.column].str.contains(code, na=False).astype(int)
        
        X[f'is_{self.column}_Other'] = (~X[[f'is_{self.column}_{item}' for item in code_column]].any(axis=1)).astype(int)
        
        self.cols = [f'is_{self.column}_{i}' for i in code_column]
        self.cols.append(f'is_{self.column}_Other')
        
        return X.drop(columns=self.column)

    def get_feature_names_out(self, input_features = None):
        return self.cols


class OneHotEncoderForMultiStrFeature(BaseEstimator, TransformerMixin):
    def __init__(self, str_feature_name: str, delimiter: str = ",", skip_values: list = [] , enable_top_n: int = False) -> None:
        super().__init__()
        self.str_feature_name = str_feature_name
        self.delimiter = delimiter
        self.skip_values = skip_values
        self.features_names_out = []
        self.enable_top_n = enable_top_n

    def _get_top_n(self,X):
        code_column = pd.DataFrame(X[self.str_feature_name].str.split(self.delimiter).tolist()).stack()
        code_column = code_column.value_counts().head(self.enable_top_n).index.tolist()
        return code_column

    def _create_binary_features_from_str_feature(self, row: pd.Series, to_code = []) -> pd.Series:
        values = row[self.str_feature_name].split(self.delimiter)
        for v in values:
            if v in self.skip_values: continue
            if (not self.enable_top_n) or v in to_code:
                row['is_{0}_{1}'.format(*[self.str_feature_name,v])] = 1
            else:
                row[f'is_{self.str_feature_name}_Other'] = 1
        return row    
    

    def one_hot_encoding_for_multi_str_feature(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        to_code = self._get_top_n(data_frame) if self.enable_top_n else []

        df = data_frame.apply(lambda row: self._create_binary_features_from_str_feature(row, to_code=to_code), 
                              axis=1).drop(columns = self.str_feature_name)
        self.features_names_out = df.filter(regex=f"is_{self.str_feature_name}").columns.values.tolist()
        df[self.features_names_out]= df[self.features_names_out].fillna(0).astype(np.int64)
        return df

    def fit(self,X, y=None):
        return self
    
    def transform(self,X):
        return self.one_hot_encoding_for_multi_str_feature(X)
    
    def get_feature_names_out(self, input_features = None):
        return self.features_names_out
    
class DateSplitter(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self,X, y=None):
        return self
    
    def transform(self,X):
        X["production_date"] = pd.to_datetime(X["production_date"])
        X["production_year"], X["production_month"] = X["production_date"].dt.year, X["production_date"].dt.month
        return X.drop(columns = "production_date")
    
    def get_feature_names_out(self, input_features = None):
        return ["production_year", "production_month"]
    
class DirectorsAgeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self,X, y=None):
        return self
    
    def transform(self,X):
        X["director_birthYear"] = pd.to_numeric(X["director_birthYear"])
        X["production_date"] = pd.to_datetime(X["production_date"])

        X["director_age"] = X["production_date"].dt.year- X["director_birthYear"]

        #X["director_years_since_death"] = X["director_deathYear"].apply(lambda x: 0 if x == "alive" else x).astype(dtype=np.int64)
        #X.loc[X["director_years_since_death"] != 0,"director_years_since_death"] = 2024 - X.loc[X["director_years_since_death"] != 0,"director_years_since_death"]
        X["director_is_alive"] = X["director_deathYear"].apply(lambda x: 1 if x == "alive" else 0).astype(dtype=np.int64)
        return X.drop(columns = ["director_birthYear", "director_deathYear", "production_date"])

    def get_feature_names_out(self, input_features = None):
        return ["director_age", "director_is_alive"]

class ContinuationFinder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, X):
        pattern = r":| \d+$"
        X["is_continuation"] = X['movie_title'].apply(lambda x: 1 if re.search(pattern, x) else 0)
        return X.drop(columns = 'movie_title')
    
    def fit(self,X, y=None):
        return self
    
    def get_feature_names_out(self, input_features = None):
        return ["is_continuation"]
    
class DirectorEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, X):
        directors = X['director_name'].unique()
        director_code = {director: i for i, director in enumerate(directors)}
        X["director_code"] = X['director_name'].map(director_code)
        return X.drop(columns = 'director_name')
    
    def fit(self,X, y=None):
        return self
    
    def get_feature_names_out(self, input_features = None):
        return ["director_code"]



