
from src.preprocessing.DatasetPreprocessor import DatasetPreprocessor
from src.targetvar.TargetVariableCreator import TargetVariableCreator
from src.features.numeric.NumericValueNullCreator import NumericValueNullCreator
from src.features.verbose.VerboseValueNullBinaryCreator import VerboseValueNullBinaryCreator

from src.features.categorical.CategoricalValueNullBinaryNominalCreator import CategoricalValueNullBinaryNominalCreator


class DatasetBuilder(DatasetPreprocessor):
    def __init__(self, dataset, target_var='', cont_num_columns=[], discrete_num_columns=[], nominal_cat_columns=[], verbose_columns=[], verbose_threshold=[], verbose_most_common=[]):
        """PreProcess Training and Testing Data."""
        super().__init__(dataset, target_var, cont_num_columns, discrete_num_columns,
                         nominal_cat_columns, verbose_columns, verbose_threshold, verbose_most_common)

    def build_model_input(self):
        """Run ETL of the column."""
        self._create_target_variable()
        self._create_numeric_features()
        # self._create_verbose_features()
        # self._create_custom_categorical_nominal_features()
        # self._create_categorical_nominal_features()
        # self._target_var_greater_than_zero()
        # self._train_valid_test_split()
        # # self._scale_train_valid_test()