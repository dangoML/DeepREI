from src.features.numeric.NumericValueNull import NumericValueNull
import pandas as pd


class NumericValueNullCreator():
    def __init__(self, cont_num_columns=[], discrete_num_columns=[]):
        """Create Numeric Features."""

        self.cont_num_columns = cont_num_columns
        self.discrete_num_columns = discrete_num_columns

    def _create_numeric_features(self):
        """ETL our Numeric Features."""

        # Loop through each column and create numeric feature
        for column in self.cont_num_columns:
            NumericValueNull(self.dataset[column], name=column).run_etl()

        for column in self.discrete_num_columns:
            NumericValueNull(self.dataset[column], name=column).run_etl()
