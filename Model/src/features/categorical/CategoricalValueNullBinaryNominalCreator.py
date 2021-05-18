
from src.features.categorical.CategoricalValueNullBinaryNominal import CategoricalValueNullBinaryNominal
from src.features.categorical.nominal_custom.CategoricalValueNullBinaryNominalCustom_area import CategoricalValueNullBinaryNominalCustom_area
import pandas as pd


class CategoricalValueNullBinaryNominalCreator():
    def __init__(self, nominal_cat_columns=[]):
        """Create Categorical Nominal Features."""

        # Inputs
        self.nominal_cat_columns = nominal_cat_columns

        # Outputs
        self.cat_custom_nominals = []

    def _create_custom_categorical_nominal_features(self):
        """Create custom categorical nominal features and add to Model Input DF."""

        # Area Custom Handler
        CategoricalValueNullBinaryNominalCustom_area(
            self.dataset['area'], name='area').run_etl()
        self.cat_custom_nominals = self.cat_custom_nominals + ['area']

    def _create_categorical_nominal_features(self):
        """Create categorical nominal features and add to Model Input DF."""

        # Create list of all columns that didn't have a custom handler
        remaining_cat_columns = set(self.nominal_cat_columns).difference(
            set(self.cat_custom_nominals))

        # Loop through each column and create categorical feature
        for column in remaining_cat_columns:
            CategoricalValueNullBinaryNominal(
                self.dataset[column], name=column).run_etl()
