from src.eda.multivariate_feature_analysis.ContTargetContFeatureAnalyzer import ContTargetContFeatureAnalyzer
from src.eda.multivariate_feature_analysis.ContTargetCatFeatureAnalyzer import ContTargetCatFeatureAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt


class MultivariateFeatureAnalyzer():
    def __init__(self, dataset=None, is_target_continuous=True, is_feature_continuous=True, target_col_name='',
                 feature_col_names=[], suffix=""):
        """Bivariate Statistics and Visualizations for Continuous/Categorical Features."""

        # Inputs
        self.is_target_continuous = is_target_continuous
        self.is_feature_continuous = is_feature_continuous
        self.dataset = dataset
        self.target_col_name = target_col_name
        self.feature_col_names = feature_col_names
        self.suffix = suffix

        # Run EDA
        if self.is_target_continuous == True and is_feature_continuous == True:
            eda = ContTargetContFeatureAnalyzer(dataset=self.dataset, target_col_name=self.target_col_name,
                                                suffix=self.suffix, feature_col_names=self.feature_col_names)
            eda.run_eda()

            # Run EDA
        if self.is_target_continuous == True and is_feature_continuous == False:
            eda = ContTargetCatFeatureAnalyzer(dataset=self.dataset, target_col_name=self.target_col_name,
                                               suffix=self.suffix, feature_col_names=self.feature_col_names)
            eda.run_eda()
