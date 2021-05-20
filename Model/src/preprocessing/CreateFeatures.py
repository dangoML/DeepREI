from src.features.VerboseValueNullBinary import VerboseValueNullBinary
from src.features.NumericValueNullBinary import NumericValueNullBinary
from src.features.CategoricalValueNullBinary import CategoricalValueNullBinary
from src.features.ContinuousTargetVariable import ContinuousTargetVariable

import pandas as pd


class CreateFeatures():
    def __init__(self, dataset, target_var, num_columns, cat_columns, verbose_columns, verbose_threshold, verbose_most_common):
        """Create Features from Raw Data."""
        # Inputs
        self.dataset = dataset[dataset['listingtype'] == 'sold']
        self.target_var = target_var
        self.target_cont_column = self.dataset[self.target_var]
        self.num_columns = num_columns
        self.cat_columns = cat_columns
        self.verbose_columns = verbose_columns
        self.verbose_threshold = verbose_threshold
        self.verbose_most_common = verbose_most_common

        # Outputs
        self.df_model = pd.DataFrame()

    def _create_target_variable(self):
        """ETL our Continuous Target Variable."""

        # Loop through each column and create numeric feature
        target = ContinuousTargetVariable(self.target_cont_column)
        target.run_etl()
        self.df_model = pd.concat([self.df_model, target.df_etl], axis=1)

    def _create_numeric_features(self):
        """Create numeric features and add to Model Input DF."""

        # Loop through each column and create numeric feature
        for column in self.num_columns:
            numeric = NumericValueNullBinary(self.dataset[column])
            numeric.run_etl()
            self.df_model = pd.concat([self.df_model, numeric.df_etl], axis=1)

    def _create_verbose_features(self):
        """Create verbose features and add to Model Input DF."""

        # Loop through each column and create verbose feature
        for column in self.num_columns:
            verbose = VerboseValueNullBinary(
                self.dataset[column], threshold=self.verbose_threshold, most_common=self.verbose_most_common)
            verbose.run_etl()
            self.df_model = pd.concat([self.df_model, verbose.df_etl], axis=1)

    def _create_categorical_features(self):
        """Create categorical features and add to Model Input DF."""

        # Loop through each column and create categorical feature
        for column in self.num_columns:
            categorical = CategoricalValueNullBinary(self.dataset[column])
            categorical.run_etl()
            self.df_model = pd.concat(
                [self.df_model, categorical.df_etl], axis=1)
