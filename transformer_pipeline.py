from sklearn.compose import ColumnTransformer
from custom_preprocessors import OneHotEncoderForMultiStrFeature, DateSplitter, DirectorsAgeTransformer, ContinuationFinder, DirectorEncoder, StringiEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import PowerTransformer, RobustScaler

def drop_rows(df):
    return df.loc[(df["director_name"] != "-") & 
                  (df["director_professions"] != "-") & 
                  (df["director_birthYear"] != "\\N"), :]


def get_non_numeric_features_transformer():
    COLS_TO_DROP = ['movie_numerOfVotes',
                    'movie_averageRating', 
                    'Domestic gross $',
                    'director_name', #we will see if we will drop this
                    ]
    return ColumnTransformer(
        transformers= [
            ("genres_OHE", OneHotEncoderForMultiStrFeature("genres",skip_values=["\\N","News"], enable_top_n=5), ["genres"]),
            ("director_professions_OHE", OneHotEncoderForMultiStrFeature("director_professions", enable_top_n=3), ["director_professions"]),
            ("production_date_split", DateSplitter(), ["production_date"]),
            ("directors_age", DirectorsAgeTransformer(), ["director_birthYear", "director_deathYear", "production_date"]),
            # ('director', DirectorEncoder(), ['director_name']), # we will see if we will drop this
            ('is_continuation', ContinuationFinder(), ['movie_title']),
            ("drop_columns", 'drop', COLS_TO_DROP)
        ],
        remainder= "passthrough"
    )

def get_numeric_features_transformer(gross_transformer, budget_transformer, runtime_transformer, approval_transformer):
    return ColumnTransformer(
        transformers= [
            ("WW_gross_$_box_cox", gross_transformer, ['Worldwide gross $']),
            ('Production_budget', budget_transformer, ['Production budget $']),
            ("runtime_minutes_Robust", runtime_transformer, ["runtime_minutes"]),
            ("approval_index", approval_transformer, ["approval_Index"])
        ],
        remainder = "passthrough"
    )

def get_features_transformer(gross_transformer: ColumnTransformer=StandardScaler(), 
                             budget_transformer: ColumnTransformer=StandardScaler(), 
                             runtime_transformer: ColumnTransformer=StandardScaler(), 
                             approval_transformer: ColumnTransformer=StandardScaler()):
    for_non_numeric = ['movie_numerOfVotes',
                    'movie_averageRating', 
                    'Domestic gross $', 
                    "movie_title", 
                    'director_name', #do not delete this
                    "genres",
                    "director_professions",
                    "production_date",
                    "director_birthYear", "director_deathYear"]
    for_numeric = ['Worldwide gross $',"runtime_minutes","approval_Index",'Production budget $']

    return ColumnTransformer(
        transformers= [ 
            ("non_numeric",  get_non_numeric_features_transformer(), for_non_numeric),
            ("numeric", get_numeric_features_transformer(gross_transformer, budget_transformer, runtime_transformer, approval_transformer), for_numeric)
        ]
    )
