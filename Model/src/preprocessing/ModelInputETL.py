from src.features.numeric.ContinuousNumericFeatureCreator import ContinuousNumericFeatureCreator
from src.features.numeric.DiscreteNumericFeatureCreator import DiscreteNumericFeatureCreator
from src.features.targetvar.TargetVariableCreator import TargetVariableCreator
from src.features.categorical.CategoricalFeatureCreator import CategoricalFeatureCreator
from src.features.verbose.VerboseFeatureCreator import VerboseFeatureCreator
import pandas as pd


class ModelInputETL():
    def __init__(self, dataset, target_var='', cont_num_columns=[], discrete_num_columns=[], nominal_cat_columns=[], 
                        verbose_columns=[], verbose_threshold=[], verbose_most_common=[], pca_columns=[], pca_expl_var=.95):
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
        self.pca_columns = pca_columns
        self.pca_expl_var = pca_expl_var

        # Outputs
        self.df_model = pd.DataFrame()

    def _etl_dataset(self):
        """Run ETL on Dataset."""
        # ETL Target Variable
        print('Performing ETL: Target Variable')
        target = TargetVariableCreator(self.dataset,self.target_var)
        target.run_feature_etl()
    
        # ETL Continuous Numeric Features
        print('Performing ETL: Continuous Numeric Features')
        contnumeric = ContinuousNumericFeatureCreator(self.dataset, cont_num_columns=self.cont_num_columns)
        contnumeric.run_feature_etl()

        # ETL Discrete Numeric Features
        print('Performing ETL: Discrete Numeric Features')
        discretenumeric = DiscreteNumericFeatureCreator(self.dataset, discrete_num_columns=self.discrete_num_columns)
        discretenumeric.run_feature_etl()

        # ETL Categorical Features
        print('Performing ETL: Categorical Features')
        categorical = CategoricalFeatureCreator(self.dataset,self.nominal_cat_columns)
        categorical.run_feature_etl()

        # ETL Verbose Features
        print('Performing ETL: Verbose Features')
        verbose = VerboseFeatureCreator(self.dataset,verbose_columns=self.verbose_columns, verbose_threshold=self.verbose_threshold, verbose_most_common=self.verbose_most_common)
        verbose.run_feature_etl()

        # Return Model Input DF
        self.df_model = pd.concat([target.df_etl,contnumeric.df_etl,discretenumeric.df_etl,categorical.df_etl,verbose.df_etl],axis=1)

        # Remove variables from memory
        del target
        del contnumeric
        del discretenumeric
        del categorical
        del verbose