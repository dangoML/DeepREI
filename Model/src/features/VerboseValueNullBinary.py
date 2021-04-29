from src.features.feature import Feature
from collections import Counter
import re
import pandas as pd


class VerboseValueNullBinary(Feature):
    def __init__(self, col_series, threshold, most_common):
        """Instantiate Verbose Feature."""
        super().__init__(col_series)  # Super?
        self.threshold = threshold
        self.most_common = most_common

    def run_etl(self):
        """Run ETL of the column."""
        self._replace_nans()
        self._add_is_null_column()
        self._string_clean_up()
        self._word_count_catergorical()

    def _cast(self):
        """Cast all values in the column to ensure correct type."""
        self.col_etl = self.col_etl.astype(float)

    def _add_is_null_column(self):
        """Create an is_null column."""
        # Grab indexes of null columns
        null_indexes = pd.isna(self.col_etl)

        # Create 'null' column variant names
        null_key = "_".join([self.name, "is_null"])

        # Create Binary 'null' column variants in DataFrame
        self.df_etl[null_key] = self.col_etl.fillna(1)
        self.df_etl[null_key].loc[~null_indexes] = 0

    def _string_clean_up(self):
        def _clean_up(row):
            try:
                clean = re.sub(r'[^\w\s]', '', row.lower()).split(' ')
                return clean

            except:
                # Run it and see if there are many rows with parse error
                return ['none']

        # Lower case string, Remove Punctuation, and Split on ' '
        self.col_etl = self.col_etl.apply(lambda x: _clean_up(x))

    def _word_count_catergorical(self):
        """Create Catergorical columns based on word count thresholds and
        whether or not the current observation contains said word."""

        # Count all instances of a word
        results = Counter()
        self.col_etl.apply(results.update)

        # Make list of words over threshold if most_common = True
        if self.most_common:
            word_feats = [(x) for (x, y) in results.most_common()
                          if y > self.threshold]

        # Else Make list of words below threshold if most_common = False
        else:
            word_feats = [(x) for (x, y) in results.most_common()
                          if y <= self.threshold]

        # For Each word in list, make column and assign Binary value
        for word in word_feats:

            # Return binary, whether iteration is in list of strings
            self.df_etl[f'{self.col_etl.name}_cat_{word}'] = self.col_etl.apply(
                lambda x: 1 if word in x else 0)
