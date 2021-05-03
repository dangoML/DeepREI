from src.features.VerboseValueNullBinary import VerboseValueNullBinary
from src.features.NumericValueNullBinary import NumericValueNullBinary
from src.features.CategoricalValueNullBinary import CategoricalValueNullBinary
from collections import Counter
import re
import pandas as pd


class BuildModelInput():
    def __init__(self, dataset, num_columns, cat_columns, verbose_columns, verbose_threshold, verbose_most_common):
        """Instantiate Model Input Table."""
        # Inputs
        self.dataset = dataset
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.verbose_columns = verbose_columns
        self.verbose_threshold = verbose_threshold
        self.verbose_most_common = verbose_most_common

        # Outputs
        self.df_model = pd.DataFrame()

    def build_model_input(self):
        """Run ETL of the column."""
        self._create_verbose_features()
        self._create_categorical_features()
        self._create_numeric_features()

    def _create_verbose_features(self):
        """Create verbose features and add to Model Input DF."""
        for column in self.num_columns:
            verbose = VerboseValueNullBinary(
                self.dataset[column], threshold=self.verbose_threshold, most_common=self.verbose_most_common)
            verbose.run_etl()
            self.df_model = pd.concat([self.df_model, verbose.df_etl], axis=1)

    def _create_categorical_features(self):
        """Create categorical features and add to Model Input DF."""
        for column in self.num_columns:
            categorical = CategoricalValueNullBinary(self.dataset[column])
            categorical.run_etl()
            self.df_model = pd.concat(
                [self.df_model, categorical.df_etl], axis=1)

    def _create_numeric_features(self):
        """Create numeric features and add to Model Input DF."""
        for column in self.num_columns:
            numeric = NumericValueNullBinary(self.dataset[column])
            numeric.run_etl()
            self.df_model = pd.concat([self.df_model, numeric.df_etl], axis=1)
