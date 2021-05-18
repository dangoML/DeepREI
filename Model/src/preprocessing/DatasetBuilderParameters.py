import pandas as pd


class DatasetBuilderParameters():
    def __init__(self, dataset, verbose_columns, verbose_threshold, verbose_most_common):
        """Create Features from Raw Data."""
        # Inputs
        self.dataset = dataset[dataset['listingtype'] == 'sold']

        self.verbose_columns = verbose_columns
        self.verbose_threshold = verbose_threshold
        self.verbose_most_common = verbose_most_common

        # Outputs
        self.df_model = pd.DataFrame()
