from src.features.targetvar.TargetVariableContinuous import TargetVariableContinuous

import pandas as pd


class TargetVariableCreator():
    def __init__(self, target_var):
        """Create Target Variable."""
        
        # self.dataset = dataset
        self.target_var = target_var
        self.target_cont_column = self.dataset[self.target_var] # How Can I Pull in Dataset to here???

    def _create_target_variable(self):
        """ETL our Continuous Target Variable."""

        TargetVariableContinuous(
            self.target_cont_column, name=self.target_var).run_etl()
