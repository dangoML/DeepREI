from src.features.target.TargetVariableContinuous import TargetVariableContinuous

import pandas as pd


class TargetVariableCreator():
    def __init__(self, dataset, target_var):
        """Create Target Variable."""
        # self.dataset = dataset
        self.target_var = target_var
        self.target_cont_column = self.dataset[self.target_var]

    def _create_target_variable(self):
        """ETL our Continuous Target Variable."""

        TargetVariableContinuous(
            self.target_cont_column, name=self.target_var).run_etl()
