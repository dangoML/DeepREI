from src.features.feature import Feature
import pandas as pd


class NumericValueNullBinary(Feature):
    def __init__(self, col_series):
        """Instantiate new dollar value feature."""
        super().__init__(col_series)  # Super?

    def run_etl(self):
        """Run ETL of the column."""
        self._replace_nans()
        self._add_value_column_df()
        self._add_is_null_column_df()
        self._build_etl_df()
