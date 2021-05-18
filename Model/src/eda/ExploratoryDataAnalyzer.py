from src.eda.BivariateFeatureAnalyzer import BivariateFeatureAnalyzer


class ExploratoryDataAnalyzer(BivariateFeatureAnalyzer):
    def __init__(self, target_var_series=None, target_var_is_cont=True, feature_is_cont=True,  feature_df=None, feature_col_name='', suffix="", bins=100, x_axis_limit=None):
        """Creator a colection of Summary Statistics and Visualizations for Continuous and Categorical Features."""
        super().__init__(target_var_series, target_var_is_cont, feature_is_cont, feature_df, feature_col_name,
                         suffix, bins, x_axis_limit)

        # Run Methods
        self._continuous_univariate_statistics()
        self._continuous_univariate_visualizations()
        self._bivariate_analyzer()
