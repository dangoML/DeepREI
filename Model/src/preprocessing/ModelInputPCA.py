from src.preprocessing.ModelInputPreprocessor import ModelInputPreprocessor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class ModelInputPCA(ModelInputPreprocessor):
    def __init__(self, dataset, target_var='', cont_num_columns=[], discrete_num_columns=[], nominal_cat_columns=[], 
                        verbose_columns=[], verbose_threshold=[], verbose_most_common=[], pca_columns=[], pca_expl_var=.95):
        """Instantiate Model Input Table."""
        super().__init__(dataset, target_var, cont_num_columns, discrete_num_columns,
                         nominal_cat_columns, verbose_columns, verbose_threshold, verbose_most_common, pca_columns, pca_expl_var)

    def _pca_transform(self):
        """PCA Transfrom Columns."""

        for pca_column in self.pca_columns:
            # Return DF filtere on orig columns
            orig_columns = [x for x in self.df_model.columns if pca_column+'_' in x and '_is_null' not in x]
            orig_columns_train = self.df_X_train[orig_columns]
            orig_columns_valid = self.df_X_valid[orig_columns]
            orig_columns_test = self.df_X_test[orig_columns]

            # PCA Fit-Transform
            pca = PCA(self.pca_expl_var)
            pca.fit(orig_columns_train)
            new_columns_train = pd.DataFrame(pca.transform(orig_columns_train))
            new_columns_valid = pd.DataFrame(pca.transform(orig_columns_valid))
            new_columns_test = pd.DataFrame(pca.transform(orig_columns_test))

            # Drop original pre-PCA columns
            self.df_X_train = self.df_X_train.drop(orig_columns,axis=1)
            self.df_X_valid = self.df_X_valid.drop(orig_columns,axis=1)
            self.df_X_test = self.df_X_test.drop(orig_columns,axis=1)

            # Drop original pre-PCA columns
            new_columns_train.columns = [pca_column + '_pca_' + str(col) for col in new_columns_train.columns]
            new_columns_valid.columns = [pca_column + '_pca_' + str(col) for col in new_columns_valid.columns]
            new_columns_test.columns = [pca_column + '_pca_' + str(col) for col in new_columns_test.columns]

            # Combine PCA df with train,valid,test dfs
            self.df_X_train = pd.concat([self.df_X_train,new_columns_train],axis=1)
            self.df_X_valid = pd.concat([self.df_X_valid,new_columns_valid],axis=1)
            self.df_X_test = pd.concat([self.df_X_test,new_columns_test],axis=1)