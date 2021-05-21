from src.features.numeric.NumericValueNull import NumericValueNull
import pandas as pd


class NumericFeatureCreator():
    def __init__(self, dataset, cont_num_columns=[], discrete_num_columns=[]):
        """Create Numeric Features."""
        # Inputs
        self.dataset = dataset
        self.cont_num_columns = cont_num_columns
        self.discrete_num_columns = discrete_num_columns

        # Outputs
        self.df_etl = pd.DataFrame()

    def _create_numeric_features(self):
        """ETL our Numeric Features."""
        # Loop through each column and create numeric feature
        if self.cont_num_columns is not None:
            for column in self.cont_num_columns:
                numeric = NumericValueNull(self.dataset[column])
                numeric.run_etl()
                self.df_etl = pd.concat([self.df_etl, numeric.df_etl], axis=1)
        
        if self.discrete_num_columns is not None:
            for column in self.discrete_num_columns:
                numeric = NumericValueNull(self.dataset[column])
                numeric.run_etl()
                self.df_etl = pd.concat([self.df_etl, numeric.df_etl], axis=1)
