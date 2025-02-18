import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datasets import create_cache_directory
from datasets import PATH as DATASET_PATH


DEFAULT_SEED = 0
DEFAULT_TEST_SIZE = 1/3
DEFAULT_VALIDATION_SIZE = 1/3
DEFAULT_COMPAS_DATASET_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"


class CompasLoader:
    class CompasProcessor:
        def __init__(self, seed: int = DEFAULT_SEED):
            self.seed = seed

        def split(
            self,
            df: pd.DataFrame,
            validation_size: float = DEFAULT_VALIDATION_SIZE,
            test_size: float = DEFAULT_TEST_SIZE,
            validation: bool = True,
        ) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame] or [pd.DataFrame, pd.DataFrame]:
            train_df, test_df = train_test_split(
                df, test_size=test_size, stratify=df["two_year_recid"], random_state=self.seed
            )
            if not validation:
                return train_df, test_df
            else:
                train_df, val_df = train_test_split(
                    train_df,
                    test_size=validation_size,
                    stratify=train_df["two_year_recid"],
                    random_state=self.seed,
                )
                return train_df, val_df, test_df

        def setup(
            self,
            df: pd.DataFrame,
            preprocess: bool = True,
            min_max: bool = False,
        ) -> pd.DataFrame:
            output = df.pop("two_year_recid")
            # Date to int
            for column in CompasLoader.date:
                df[column] = pd.to_datetime(df[column]).astype(int)
            # Remove personal information
            for column in CompasLoader.personal:
                df.drop([column], axis=1, inplace=True)
            # Remove all columns with one unique value (no information)
            for column in df.columns:
                if len(df[column].unique()) == 1:
                    df.drop([column], axis=1, inplace=True)
            # missing and nan to 0
            df.fillna(0, inplace=True)
            for column in CompasLoader.categorical:
                df[column] = df[column].astype("category").cat.codes
            if preprocess:
                scaler = StandardScaler() if not min_max else MinMaxScaler()
                df = pd.DataFrame(scaler.fit_transform(df))
            df["two_year_recid"] = output
            return df

    filename = "compas.csv"
    processor = CompasProcessor()

    personal = [
        "id",
        "name",
        "first",
        "last",
    ]
    date = [
        "compas_screening_date",
        "dob",
        "c_jail_in",
        "c_jail_out",
        "c_offense_date",
        "c_arrest_date",
        "r_offense_date",
        "r_jail_in",
        "r_jail_out",
        "vr_offense_date",
        "screening_date",
        "v_screening_date",
        "in_custody",
        "out_custody",
    ]
    categorical = [
        "sex",
        "age_cat",
        "race",
        "c_case_number",
        "r_case_number",
        "c_charge_degree",
        "c_charge_desc",
        "r_charge_degree",
        "r_charge_desc",
        "vr_case_number",
        "vr_charge_degree",
        "vr_charge_desc",
        "score_text",
        "v_score_text",

    ]

    def __init__(self, path: str = DEFAULT_COMPAS_DATASET_URL):
        self.path = path

    def load(self, url: str = None) -> pd.DataFrame:
        if url is None:
            url = self.path
        create_cache_directory()
        cache_file = DATASET_PATH / "cache" / ("compas.csv")
        if cache_file.exists():
            return pd.read_csv(cache_file)
        else:
            df = pd.read_csv(url, skipinitialspace=True, header=0)
            df.to_csv(cache_file, index=False)
            return df

    def load_all(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = self.load()
        df_train, df_test = train_test_split(df, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_SEED, stratify=df["two_year_recid"])
        return df_train, df_test

    def load_preprocessed(
        self,
        all_datasets: bool = False,
        preprocess: bool = True,
        min_max: bool = False,
    ) -> pd.DataFrame or tuple[pd.DataFrame, pd.DataFrame]:
        if all_datasets:
            df_train, df_test = self.load_all()
            df = pd.concat([df_train, df_test], axis=0)
            df = self.processor.setup(df, preprocess=preprocess, min_max=min_max)
            train, test = df.iloc[:len(df_train), ], df.iloc[len(df_train):, ]
            return train, test
        else:
            df = self.load()
            return self.processor.setup(df, preprocess=preprocess, min_max=min_max)

    def load_preprocessed_split(
        self,
        validation: bool = True,
    ) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame] or [pd.DataFrame, pd.DataFrame]:
        df = self.load_preprocessed()
        return self.processor.split(df, validation=validation)
