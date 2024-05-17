import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class OneHotEncoderForMultiStrFeature(BaseEstimator, TransformerMixin):
    def __init__(self, str_feature_name: str, delimiter: str = ",", skip_values: list = [] ) -> None:
        super().__init__()
        self.str_feature_name = str_feature_name
        self.delimiter = delimiter
        self.skip_values = skip_values

        self.features_names_out = []

    def _create_binary_features_from_str_feature(self, row: pd.Series) -> pd.Series:
        values = row[self.str_feature_name].split(self.delimiter)
        for v in values:
            if v in self.skip_values: continue
            row['is_{0}_{1}'.format(*[self.str_feature_name,v])] = 1
        return row

    def one_hot_encoding_for_multi_str_feature(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        df = data_frame.apply(lambda row: self._create_binary_features_from_str_feature(row), 
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

        X["director_years_since_death"] = X["director_deathYear"].apply(lambda x: 0 if x == "alive" else x).astype(dtype=np.int64)
        return X.drop(columns = ["director_birthYear", "director_deathYear", "production_date"])

    def get_feature_names_out(self, input_features = None):
        return ["director_age", "director_years_since_death"]





