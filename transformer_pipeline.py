from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import PowerTransformer, RobustScaler
from custom_preprocessors import OneHotEncoderForMultiStrFeature, DateSplitter, DirectorsAgeTransformer, ContinuationFinder, DirectorEncoder


def drop_rows(df):
    return df.loc[(df["director_name"] != "-") & 
                  (df["director_professions"] != "-") & 
                  (df["director_birthYear"] != "\\N"), :]


def get_non_numeric_features_transformer():
    COLS_TO_DROP = ['movie_numerOfVotes',
                    'movie_averageRating', 
                    'Domestic gross $'
                    ]
    return ColumnTransformer(
        transformers= [
            ("genres_OHE", OneHotEncoderForMultiStrFeature("genres",skip_values=["\\N","News"]), ["genres"]),
            ("director_professions_OHE", OneHotEncoderForMultiStrFeature("director_professions", skip_values=['casting_director', 'make_up_department']), ["director_professions"]),
            ("production_date_split", DateSplitter(), ["production_date"]),
            ("directors_age", DirectorsAgeTransformer(), ["director_birthYear", "director_deathYear", "production_date"]),
            ('director_code', DirectorEncoder(), ['director_name']),
            ('is_continuation', ContinuationFinder(), ['movie_title']),
            ("drop_columns", 'drop', COLS_TO_DROP)
        ],
        remainder= "passthrough"
    )

def get_numeric_features_transformer():
    return ColumnTransformer(
        transformers= [
            ("WW_gross_$_box_cox", PowerTransformer(method="box-cox"), ['Worldwide gross $']),
            ('Production_budget', PowerTransformer(method="box-cox"), ['Production budget $']), #na pałę narazie
            ("runtime_minutes_Robust", RobustScaler(), ["runtime_minutes"]),
            ("approval_index", StandardScaler(), ["approval_Index"])
        ],
        remainder = "passthrough"
    )

def get_features_transformer():
    for_non_numeric = ['movie_numerOfVotes',
                    'movie_averageRating', 
                    'Domestic gross $', 
                    "movie_title", 
                    'director_name',
                    "genres",
                    "director_professions",
                    "production_date",
                    "director_birthYear", "director_deathYear"]
    for_numeric = ['Worldwide gross $',"runtime_minutes","approval_Index", 'Production budget $']

    return ColumnTransformer(
        transformers= [ 
            ("non_numeric",  get_non_numeric_features_transformer(), for_non_numeric),
            ("numeric", get_numeric_features_transformer(), for_numeric)
        ]
    )
