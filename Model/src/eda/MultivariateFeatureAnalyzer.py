from src.eda.UnivariateFeatureAnalyzer import UnivariateFeatureAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt


class MultivariateFeatureAnalyzer():
    def __init__(self, is_target_continuous=True, is_feature_continuous=True, dataset=None, target_col_name='', feature_col_names=[], suffix=""):
        """Bivariate Statistics and Visualizations for Continuous/Categorical Features."""

        # Inputs
        self.is_target_continuous = is_target_continuous
        self.is_feature_continuous = is_feature_continuous
        self.dataset = dataset
        self.target_col_name = target_col_name
        self.feature_col_names = feature_col_names
        self.suffix = suffix

        if self.is_target_continuous == True and self.is_feature_continuous == True:
            self._cont_target_vs_cont_features()

    def _cont_target_vs_cont_features(self):

        # Create list of Feature Names + Suffix
        num_columns_value = [x+self.suffix for x in self.feature_col_names]

        # Plot Target vs all Features
        g = sns.pairplot(data=self.dataset,
                         y_vars=self.target_col_name,
                         x_vars=num_columns_value)
        g.fig.suptitle(
            f"{self.target_col_name} vs Continuous Features", y=1.08)

        # get correlation matrix
        num_columns_data = self.dataset[[
            self.target_col_name]+num_columns_value]
        num_columns_data = num_columns_data.corr().abs()

        # draw the heatmap using seaborn.
        plt.figure(figsize=(10, 6))
        sns.heatmap(num_columns_data, square=True, annot=True, linewidths=.5)
        plt.title("correlation matrix (Title)")
        plt.show()
