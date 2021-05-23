import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class UnivariateFeatureAnalyzer():
    def __init__(self, dataset=None, feature_type=None, feature_col_name='', suffix="", bins=100, x_axis_limit=None):
        """Univariate Visualizations and Statistics for Continuous/Categorical Features."""

        # Inputs
        # If a feature is passed, instantiate feature
        self.feature_type = feature_type
        self.dataset = dataset[feature_col_name + suffix]
        self.feature_col_name = feature_col_name + suffix
        self.bins = bins
        self.x_axis_limit = x_axis_limit

        # Outputs
        self.data_min = 0
        self.data_max = 0
        self.five_num_sum = pd.DataFrame()

    # def run(self):
        self._continuous_univariate_statistics()
        self._continuous_univariate_visualizations()

    def _continuous_univariate_statistics(self):

        if self.feature_type == True:
            # Calculate Max and Min
            self.data_min = self.dataset.min()
            self.data_max = self.dataset.max()

            if self.x_axis_limit is None:
                self.x_axis_limit = self.data_max

            # Calculate a 5-Number summary for each Continuous Varaible
            self.five_num_sum = pd.DataFrame(
                columns=['Feature', 'Min', 'Q1', 'Median', 'Q3', 'Max'])

            # calculate quartiles
            quartiles = np.percentile(self.dataset, [25, 50, 75])
            self.five_num_sum = self.five_num_sum.append(pd.DataFrame(data=[[self.feature_col_name, self.data_min, quartiles[0], quartiles[1], quartiles[2], self.data_max]],
                                                                      columns=['Feature', 'Min', 'Q1', 'Median', 'Q3', 'Max']))

            # Return results
            display(self.five_num_sum)

    def _continuous_univariate_visualizations(self):
        """Continuous univariate visualizations"""

        if self.feature_type == True:
            plt.figure(figsize=(20, 5))

            # Histogram
            plt.subplot(1, 3, 1)
            sns.histplot(self.dataset, bins=self.bins)
            plt.xlim([1, self.x_axis_limit])

            # Boxplot
            plt.subplot(1, 3, 2)
            sns.boxplot(self.dataset)
            plt.xlim([1, self.x_axis_limit])

    def _categorical_univariate_statistics(self):

        if self.feature_type == False:
            pass

    def _categorical_univariate_visualizations(self):

        if self.is_contfeature_typeinuous == False:
            pass
