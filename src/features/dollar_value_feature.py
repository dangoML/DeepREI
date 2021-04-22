from src.features.feature import Feature
import pandas as pd


class DollarValueFeature(Feature):
    def __init__(self, col_series):
        """Instantiate new dollar value feature."""
        super().__init__(col_series)

    def run_etl(self):
        """Run ETL of the column."""
        self._replace_nans()
        self._cast()
        self._add_is_null_column()

    def _cast(self):
        """Cast all values in the column to ensure correct type."""
        self.col_etl = self.col_etl.astype(float)

    def _add_is_null_column(self):
        """Create an is_null column."""
        null_indexes = pd.isna(self.col_etl)
        value_key = "_".join([self.name, "value"])
        null_key = "_".join([self.name, "is_null"])
        self.df_etl[value_key] = self.col_etl.copy().fillna(0)
        self.df_etl[null_key] = self.col_etl.copy().fillna(1)
        self.df_etl[null_key].loc[null_indexes] = 0