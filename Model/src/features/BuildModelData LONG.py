from src.features.VerboseValueNullBinary import VerboseValueNullBinary
from src.features.NumericValueNullBinary import NumericValueNullBinary
from src.features.CategoricalValueNullBinary import CategoricalValueNullBinary
from src.features.ContinuousTargetVariable import ContinuousTargetVariable
from src.features.PreProcessData import PreProcessData

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import re
import pandas as pd


class BuildModelData():
    def __init__(self, dataset, target_var, num_columns, cat_columns, verbose_columns, verbose_threshold, verbose_most_common):
        """Instantiate Model Input Table."""
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
        self.df_X_train = pd.DataFrame()
        self.df_y_train = pd.DataFrame()
        self.df_X_valid = pd.DataFrame()
        self.df_y_valid = pd.DataFrame()
        self.df_X_test = pd.DataFrame()
        self.df_y_test = pd.DataFrame()

    def build_model_input(self):
        """Run ETL of the column."""
        self._create_target_variable()
        self._create_numeric_features()
        self._create_verbose_features()
        self._create_categorical_features()
        self._train_valid_test_split()
        # self._scale_train_valid_test()

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

    def _train_valid_test_split(self):
        """Split dataset into training, validation, test"""
        # Specify X and y
        y = self.df_model[self.target_var]
        X = self.df_model.drop(self.target_var, axis=1)

        # Create Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)

        # Create Train and Validation Split
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

        # Assign splits to respective tuple objects
        self.df_X_train = X_train
        self.df_y_train = y_train
        self.df_X_valid = X_valid
        self.df_y_valid = y_valid
        self.df_X_test = X_test
        self.df_y_test = y_test

    def _scale_train_valid_test(self):
        """Scale all Ind. Variable (Features)"""
        sc = StandardScaler()
        self.df_X_train = sc.fit_transform(self.df_X_train)
        self.df_X_valid = sc.transform(self.df_X_valid)
        self.df_X_test = sc.transform(self.df_X_test)
