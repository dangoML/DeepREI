from src.features.verbose.VerboseValueNullBinary import VerboseValueNullBinary
import pandas as pd


class VerboseValueNullBinaryCreator():
    def __init__(self, verbose_columns=[], verbose_threshold=[], verbose_most_common=[]):
        """Create Features from Raw Data."""
        # Inputs
        self.verbose_columns = verbose_columns
        self.verbose_threshold = verbose_threshold
        self.verbose_most_common = verbose_most_common

    def _create_verbose_features(self):
        """Create verbose features and add to Model Input DF."""

        # Loop through each column and create verbose feature
        for column in self.verbose_columns:
            VerboseValueNullBinary(
                self.dataset[column], threshold=self.verbose_threshold, most_common=self.verbose_most_common, name=column).run_etl()
