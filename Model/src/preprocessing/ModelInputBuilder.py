from src.preprocessing.ModelInputPreprocessor import ModelInputPreprocessor

import pandas as pd

class ModelInputBuilder(ModelInputPreprocessor):
    def __init__(self, dataset, target_var='', cont_num_columns=[], discrete_num_columns=[], nominal_cat_columns=[], verbose_columns=[], verbose_threshold=[], verbose_most_common=[]):
        """Instantiate Model Input Table."""
        super().__init__(dataset, target_var, cont_num_columns, discrete_num_columns,
                         nominal_cat_columns, verbose_columns, verbose_threshold, verbose_most_common)

    def build_model_input(self):
        """Build Model Input Table."""
        self._etl_dataset()
        self._train_valid_test_split()
        # self._scale_train_valid_test()