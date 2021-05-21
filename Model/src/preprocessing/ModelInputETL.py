from src.features.numeric.NumericFeatureCreator import NumericFeatureCreator
from src.features.targetvar.TargetVariableCreator import TargetVariableCreator
from src.features.categorical.CategoricalFeatureCreator import CategoricalFeatureCreator
from src.features.verbose.VerboseFeatureCreator import VerboseFeatureCreator
import pandas as pd


class ModelInputETL():
    def __init__(self, dataset, target_var='', cont_num_columns=[], discrete_num_columns=[], nominal_cat_columns=[], verbose_columns=[], verbose_threshold=[], verbose_most_common=[]):
        """Create Features from Raw Data."""
        # Inputs
        self.dataset = dataset[dataset['listingtype'] == 'sold']
        self.target_var = target_var
        self.cont_num_columns = cont_num_columns
        self.discrete_num_columns = discrete_num_columns
        self.nominal_cat_columns = nominal_cat_columns
        self.verbose_columns = verbose_columns
        self.verbose_threshold = verbose_threshold
        self.verbose_most_common = verbose_most_common

        # Outputs
        self.df_model = pd.DataFrame()

    def _etl_dataset(self):
        """Run ETL on Dataset."""
        # ETL Target Variable
        target = TargetVariableCreator(self.dataset,self.target_var)
        target._create_target_variable()
    
        # ETL Numeric Features
        numeric = NumericFeatureCreator(self.dataset, cont_num_columns=self.cont_num_columns, discrete_num_columns=self.discrete_num_columns)
        numeric._create_numeric_features()

        # ETL Categorical Features
        categorical = CategoricalFeatureCreator(self.dataset,self.nominal_cat_columns)
        categorical._create_categorical_features()

        # ETL Verbose Features
        verbose = VerboseFeatureCreator(self.dataset,verbose_columns=self.verbose_columns, verbose_threshold=self.verbose_threshold, verbose_most_common=self.verbose_most_common)
        verbose._create_verbose_features()

        # Return Model Input DF
        self.df_model = pd.concat([target.df_etl,numeric.df_etl,categorical.df_etl,verbose.df_etl],axis=1)