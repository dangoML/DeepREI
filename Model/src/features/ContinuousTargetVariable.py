from src.features.FeatureETLEngineering import FeatureETLEngineering
import pandas as pd


class ContinuousTargetVariable(FeatureETLEngineering):
    def __init__(self, col_series):
        """Instantiate numeric feature."""
        super().__init__(col_series)

    def run_etl(self):
        """Run ETL of the column."""
        self._replace_strings_with_nans()
        # self._build_feature_etl_df()

    def _replace_strings_with_nans(self):
        """Replace stings with nans"""
        self.df_etl = pd.to_numeric(self.col_series, errors='coerce')
