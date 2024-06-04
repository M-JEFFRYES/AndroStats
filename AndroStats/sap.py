import pandas as pd
from typing import Tuple
from AndroStats.performance import CanoeAnalysis
from AndroStats import get_resource


class SemenAnalysisParameters:
    ca = CanoeAnalysis()

    def __init__(self) -> None:
        self.load_parameters()
        return

    def parameters_data_path(self) -> None:
        return get_resource("data/param_percentiles.csv")

    def load_parameters(self) -> None:
        self.df = pd.read_csv(self.parameters_data_path())
        return

    def get_abnormal_threshold(self, variable: str) -> Tuple[float, bool]:
        parameter = self.df[self.df["parameter"] == variable]
        if len(parameter) != 1:
            raise Exception(f"Check parameter name: {variable} and percentiles dataset")
        abnormal_threshold = parameter.loc[0, "abnormal_threshold"]
        is_lower = parameter.loc[0, "is_lower"]
        return (abnormal_threshold, is_lower)

    def get_threshold_data(self):
        thresholding_info = {}
        for i, row in self.df.iterrows():
            threshold = row["abnormal_threshold"]
            is_lower = row["is_lower"]
            allowable_variance = self.ca.allowable_variance(threshold)
            thresholding_info[row["parameter"]] = {"allowable_variance": allowable_variance, "threshold": threshold, "is_lower": is_lower}
        return thresholding_info
