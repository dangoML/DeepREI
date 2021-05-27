import pandas as pd
from src.features.FeatureETLEngineering import FeatureETLEngineering


class CategoricalValueNullBinaryNominal_area(FeatureETLEngineering):
    def __init__(self, col_series):
        """Instantiate categorical feature."""
        super().__init__(col_series)

    def run_etl(self):
        """Run ETL of the column."""
        self._replace_nans()
        self._add_is_null_column_df()
        self._replace_strings_with_floats()
        # self._add_value_column_df()  # ONLY INCLUDE FOR EDA PURPOSES!!
        self._create_custom_bins()
        self._one_hot_encode_df()
        self._build_feature_etl_df()

    def _replace_strings_with_floats(self):
        """Replace specific string values with float equivalent"""
        self.col_etl = self.col_etl.replace('5940 Florida Other', 5940.0)

    def _create_custom_bins(self):
        """Create Custom bins for values"""
        pass