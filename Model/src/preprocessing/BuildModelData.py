from src.preprocessing.PreProcessData import PreProcessData


class BuildModelData(PreProcessData):
    def __init__(self, dataset, target_var, num_columns, cat_columns, verbose_columns, verbose_threshold, verbose_most_common):
        """Instantiate Model Input Table."""
        super().__init__(dataset, target_var, num_columns, cat_columns,
                         verbose_columns, verbose_threshold, verbose_most_common)

    def build_model_input(self):
        """Run ETL of the column."""
        self._create_target_variable()
        self._create_numeric_features()
        self._create_verbose_features()
        self._create_categorical_features()
        self._train_valid_test_split()
        # self._scale_train_valid_test()
