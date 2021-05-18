
import pandas as pd
from src.features.FeatureETLEngineering import FeatureETLEngineering


class CategoricalValueNullBinaryNominal(FeatureETLEngineering):
    def __init__(self, col_series, name):
        """Instantiate categorical feature."""
        super().__init__(col_series, name)

    def run_etl(self):
        """Run ETL of the column."""
        self._replace_nans()
        self._add_is_null_column_df()
        self._cast_float_keep_string()
        self._add_value_column_df()  # ONLY INCLUDE FOR EDA PURPOSES!!
        self._one_hot_encode_df()
        self._build_feature_etl_df()
