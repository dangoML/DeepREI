from src.features.VerboseValueNullBinary import VerboseValueNullBinary

from src.features.NumericValueNullBinary import NumericValueNullBinary
from src.features.CategoricalValueNullBinary import CategoricalValueNullBinary

from src.features.TargetVariableContinuous import TargetVariableContinuous

from src.features.CategoricalValueNullBinaryNominal import CategoricalValueNullBinaryNominal
from src.features.CategoricalValueNullBinaryNominalCustom_area import CategoricalValueNullBinaryNominalCustom_area

import pandas as pd


class FeatureCreator():
    def __init__(self, dataset, target_var, cont_num_columns, discrete_num_columns, ordinal_cat_columns, nominal_cat_columns, verbose_columns, verbose_threshold, verbose_most_common):
        """Create Features from Raw Data."""
        # Inputs
        self.dataset = dataset[dataset['listingtype'] == 'sold']
        self.target_var = target_var
        self.target_cont_column = self.dataset[self.target_var]

        self.cont_num_columns = cont_num_columns
        self.discrete_num_columns = discrete_num_columns
        self.ordinal_cat_columns = ordinal_cat_columns
        self.nominal_cat_columns = nominal_cat_columns

        self.verbose_columns = verbose_columns
        self.verbose_threshold = verbose_threshold
        self.verbose_most_common = verbose_most_common

        # Outputs
        self.df_model = pd.DataFrame()

    def _create_target_variable(self):
        """ETL our Continuous Target Variable."""

        # Loop through each column and create numeric feature
        target = TargetVariableContinuous(self.target_cont_column)
        target.run_etl()
        self.df_model = pd.concat([self.df_model, target.df_etl], axis=1)

    def _create_numeric_features(self):
        """Create numeric features and add to Model Input DF."""

        # Loop through each column and create numeric feature
        for column in [self.discrete_num_columns + self.cont_num_columns]:
            numeric = NumericValueNullBinary(self.dataset[column])
            numeric.run_etl()
            self.df_model = pd.concat([self.df_model, numeric.df_etl], axis=1)

    def _create_categorical_nominal_features(self):
        """Create categorical nominal features and add to Model Input DF."""

        # Categorical Nominal Custom Handlers
        area = CategoricalValueNullBinaryNominalCustom_area(
            self.dataset['area']).run_etl()
        self.df_model = pd.concat(
            [self.df_model, area.df_etl], axis=1)

        # Loop through each column and create categorical feature
        for column in self.cat_columns:
            categorical = CategoricalValueNullBinary(self.dataset[column])
            categorical.run_etl()
            self.df_model = pd.concat(
                [self.df_model, categorical.df_etl], axis=1)

    def _create_verbose_features(self):
        """Create verbose features and add to Model Input DF."""

        # Loop through each column and create verbose feature
        for column in self.verbose_columns:
            verbose = VerboseValueNullBinary(
                self.dataset[column], threshold=self.verbose_threshold, most_common=self.verbose_most_common)
            verbose.run_etl()
            self.df_model = pd.concat([self.df_model, verbose.df_etl], axis=1)
