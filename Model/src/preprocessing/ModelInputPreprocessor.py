from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocessing.ModelInputETL import ModelInputETL
import pandas as pd
import numpy as np


class ModelInputPreprocessor(ModelInputETL):
    def __init__(self, dataset, target_var='', cont_num_columns=[], discrete_num_columns=[], nominal_cat_columns=[], 
                        verbose_columns=[], verbose_threshold=[], verbose_most_common=[], pca_columns=[], pca_expl_var=.95):
        """Instantiate Model Input Table."""
        super().__init__(dataset, target_var, cont_num_columns, discrete_num_columns,
                         nominal_cat_columns, verbose_columns, verbose_threshold, verbose_most_common, pca_columns, pca_expl_var)

        # Outputs
        self.df_X_train = pd.DataFrame()
        self.df_y_train = pd.DataFrame()
        self.df_X_valid = pd.DataFrame()
        self.df_y_valid = pd.DataFrame()
        self.df_X_test = pd.DataFrame()
        self.df_y_test = pd.DataFrame()

    def _drop_nan_rows(self):
        """Drop rows that have atleast 3 non-nan values."""
        # Target Value must be greater than zero
        self.dataset = self.dataset.replace(to_replace='None', value=np.nan).dropna(thresh=3)

    def _feature_limit_filters(self):
        """Filter for all property values greater than zero."""
        # Target Value must be greater than zero
        self.df_model = self.df_model[self.df_model[self.target_var] > 0]

        # Garagespaces must be less 20
        self.df_model = self.df_model[self.df_model['garagespaces_value'] < 20]

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

        # Assign splits to respective tuple objects, reset indexes
        self.df_X_train, self.df_y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
        self.df_X_valid, self.df_y_valid = X_valid.reset_index(drop=True), y_valid.reset_index(drop=True)
        self.df_X_test, self.df_y_test = X_test.reset_index(drop=True), y_test.reset_index(drop=True)

    def _scale_train_valid_test(self):
        """Scale all Ind. Variable (Features)"""
        sc = StandardScaler()
        columns = self.df_X_train.columns

        # Scale DFs
        self.df_X_train = pd.DataFrame(sc.fit_transform(self.df_X_train))
        self.df_X_valid = pd.DataFrame(sc.transform(self.df_X_valid))
        self.df_X_test = pd.DataFrame(sc.transform(self.df_X_test))
        
        # Rename colums
        self.df_X_train.columns = columns
        self.df_X_valid.columns = columns
        self.df_X_test.columns = columns