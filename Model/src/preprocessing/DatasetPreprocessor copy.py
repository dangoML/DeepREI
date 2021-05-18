from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocessing.DatasetBuilderParameters import DatasetBuilderParameters
import pandas as pd


class DatasetPreprocessor(DatasetBuilderParameters):
    def __init__(self, dataset, target_var, cont_num_columns, discrete_num_columns, ordinal_cat_columns, nominal_cat_columns, verbose_columns, verbose_threshold, verbose_most_common):
        """PreProcess Training and Testing Data."""
        super().__init__(dataset, target_var, cont_num_columns, discrete_num_columns, ordinal_cat_columns,
                         nominal_cat_columns, verbose_columns, verbose_threshold, verbose_most_common)

        # Outputs
        self.df_X_train = pd.DataFrame()
        self.df_y_train = pd.DataFrame()
        self.df_X_valid = pd.DataFrame()
        self.df_y_valid = pd.DataFrame()
        self.df_X_test = pd.DataFrame()
        self.df_y_test = pd.DataFrame()

    def _train_valid_test_split(self):
        """Split dataset into training, validation, test"""
        # Specify X and y
        y = self.df_model[self.target_var]
        X = self.df_model.drop(self.target_var, axis=1)

        # Create Test Splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)

        # Create Train and Validation Split off Train Set
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

    def _target_var_greater_than_zero(self):
        """Return rows only with Target Value Greater than 0"""
        self.df_model = self.df_model[self.df_model[self.target_var] > 0]
