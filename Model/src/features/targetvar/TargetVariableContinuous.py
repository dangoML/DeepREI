from src.features.FeatureETLEngineering import FeatureETLEngineering
import pandas as pd


class TargetVariableContinuous(FeatureETLEngineering):
    def __init__(self, col_series, name):
        """Instantiate numeric feature."""
        super().__init__(col_series, name)

    def run_etl(self):
        """Run ETL of the column."""
        self._replace_strings_with_nans()
        self._build_target_etl_df()

    def _replace_strings_with_nans(self):
        """Replace stings with nans"""
        self.df_etl = pd.to_numeric(self.col_series, errors='coerce')

    def _build_target_etl_df(self):
        """Combine all dfs into one and cast as float"""
        self.df_model = self.df_etl
        self.df_model = self.df_model.astype(float)
