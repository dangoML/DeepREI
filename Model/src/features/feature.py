import pandas as pd


class Feature:
    def __init__(self, col_series):
        """Instantiate new feature class.

        'col' must be a pandas Series object.
        """
        self.col_series = col_series
        self.name = self.col_series.name
        self.col_etl = col_series.copy()

        # Is this here because we end up creating a df as a result??
        self.df_etl = pd.DataFrame()

    def _replace_nans(self, values_to_replace=["None"]):
        """Replace specified values with None values."""
        for value in values_to_replace:
            self.col_etl.loc[self.col_etl == value] = None
