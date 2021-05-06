from src.features.FeatureETLEngineering import FeatureETLEngineering
import pandas as pd


class CategoricalValueNullBinary(FeatureETLEngineering):
    def __init__(self, col_series):
        """Instantiate categorical feature."""
        super().__init__(col_series)

    def run_etl(self):
        """Run ETL of the column."""
        self._replace_nans()
        self._add_is_null_column_df()
        self._one_hot_encode_df()
        self._build_feature_etl_df()

    def _one_hot_encode_df(self):
        """One Hot Encode and add new columns to df_etl"""
        # One-Hot-Encode all Categorical Columns
        temp_dummies_df = pd.get_dummies(
            self.col_etl, prefix=f'{self.col_etl.name}_', drop_first=True)

        # Concat One-Hot Columns to df
        self.df_etl = pd.concat([self.df_etl, temp_dummies_df], axis=1)
