from src.features.categorical.CategoricalValueNullBinaryNominal import CategoricalValueNullBinaryNominal
from src.features.categorical.nominal_custom_features.CategoricalValueNullBinaryNominalCustom_area import CategoricalValueNullBinaryNominalCustom_area
import pandas as pd


class CategoricalFeatureCreator():
    def __init__(self, dataset, nominal_cat_columns=[]):
        """Create Categorical Nominal Features."""

        # Inputs
        self.dataset = dataset
        self.nominal_cat_columns = nominal_cat_columns

        # Outputs
        self.cat_custom_nominals = []
        self.df_model = pd.DataFrame()

    def _create_categorical_features(self):
        # self._create_custom_categorical_nominal_features()
        self._create_categorical_nominal_features()
    
    
    def _create_custom_categorical_nominal_features(self):
        """Create custom categorical nominal features and add to Model Input DF."""

        # Area Custom Handler
        categorical = CategoricalValueNullBinaryNominalCustom_area(
            self.dataset['area'], name='area').run_etl()
        self.cat_custom_nominals = self.cat_custom_nominals + ['area']
        self.df_model = pd.concat(
                [self.df_model, categorical.df_etl], axis=1)

    def _create_categorical_nominal_features(self):
        """Create categorical nominal features and add to Model Input DF."""

        # Create list of all columns that didn't have a custom handler
        remaining_cat_columns = set(self.nominal_cat_columns).difference(
            set(self.cat_custom_nominals))

        # Loop through each column and create categorical feature
        print(self.nominal_cat_columns)
        for column in self.nominal_cat_columns:
            print(self.dataset[column])
            categorical2 = CategoricalValueNullBinaryNominal(self.dataset[column])
            categorical2.run_etl()

            self.df_model = pd.concat(
                [self.df_model, categorical2.df_etl], axis=1)