from src.features.feature import Feature
import pandas as pd


class CategoricalValueNullBinary(Feature):
    def __init__(self, col_series):
        """Instantiate new dollar value feature."""
        super().__init__(col_series)  # Super?

    def run_etl(self):
        """Run ETL of the column."""
        self._replace_nans()
        # self._cast()
        self._add_is_null_column()
        self._one_hot_encode()

    def _cast(self):
        """Cast all values in the column to ensure correct type."""
        self.col_etl = self.col_etl.astype(float)

    def _add_is_null_column(self):
        """Create an is_null column."""
        # Grab indexes of null columns
        # null_indexes = pd.isna(self.col_etl)

        # # Create 'value' and 'null' column variant names
        # null_key = "_".join([self.name, "is_null"])

        # # Create Binary 'value' and 'null' column variants in DataFrame
        # self.df_etl[null_key] = self.col_etl.fillna(1)
        # self.df_etl[null_key].loc[~null_indexes] = 0

    def _one_hot_encode(self):
        # One-Hot-Encode all Categorical Columns
        self.df_etl = pd.get_dummies(
            self.col_etl, prefix=f'{self.col_etl.name}_', drop_first=True)

        # DROP NON COLUMNS
