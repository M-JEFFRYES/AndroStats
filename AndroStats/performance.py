import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from AndroStats import get_resource


class CanoeAnalysis:
    def __init__(self):
        self.define_performance_canoe()
        return

    def define_performance_canoe(self) -> None:
        data = [
            {"start": 0, "end": 0, "var": 1},
            {"start": 1, "end": 1, "var": 2},
            {"start": 2, "end": 3, "var": 3},
            {"start": 4, "end": 6, "var": 4},
            {"start": 7, "end": 9, "var": 5},
            {"start": 10, "end": 13, "var": 6},
            {"start": 14, "end": 19, "var": 7},
            {"start": 20, "end": 27, "var": 8},
            {"start": 28, "end": 44, "var": 9},
            {"start": 45, "end": 55, "var": 10},
            {"start": 56, "end": 72, "var": 9},
            {"start": 73, "end": 80, "var": 8},
            {"start": 81, "end": 86, "var": 7},
            {"start": 87, "end": 90, "var": 6},
            {"start": 91, "end": 93, "var": 5},
            {"start": 94, "end": 96, "var": 4},
            {"start": 97, "end": 98, "var": 3},
            {"start": 99, "end": 99, "var": 2},
            {"start": 100, "end": 100, "var": 1},
        ]

        all_df = pd.DataFrame(data)

        d = []
        for i in range(0, 101):
            d.append(
                {
                    "value": i,
                    "variance": all_df[(all_df["start"] <= i) & (all_df["end"] >= i)]["var"].values[0],
                }
            )

        self.canoe_allowances = pd.DataFrame(d)
        self.canoe_allowances["upper"] = self.canoe_allowances["value"] + self.canoe_allowances["variance"]
        self.canoe_allowances["lower"] = self.canoe_allowances["value"] - self.canoe_allowances["variance"]
        return

    def allowable_variance(self, true_value: float | int) -> float:
        try:
            val_check = int(true_value)
            variance = self.canoe_allowances[self.canoe_allowances["value"] == val_check]["variance"].iloc[0]
        except:
            raise Exception(f"Value {val_check} not found in perfomance canoe range")
        return variance

    def prediction_within_canoe(self, true_value: float | int, pred_value: float | int, multipler: float = 1) -> bool:
        try:
            val_check = int(true_value)
            variance = self.canoe_allowances[self.canoe_allowances["value"] == val_check]["variance"].iloc[0]
        except:
            raise Exception(f"Value {val_check} not found in perfomance canoe range")

        c1 = (pred_value >= (true_value - (variance * multipler))) and (pred_value <= (true_value + (variance * multipler)))
        return c1

    def array_predictions_within_canoe(self, arr1: np.array, arr2: np.array, multipler: float = 1) -> np.array:
        if (len(arr1) < 1) or (len(arr2) < 1):
            raise Exception("Empty input array")

        if len(arr1) != len(arr2):
            raise Exception("Different length input arrays")

        c_a_1 = []
        for true_value, predicted_value in zip(arr1, arr2):
            c1 = self.prediction_within_canoe(true_value, predicted_value, multipler=multipler)
            c_a_1.append(c1)
        return c_a_1

    def calculate_canoe_performance_score(self, arr1: np.array, arr2: np.array, multipler: float = 1) -> float:
        c_a_1 = self.array_predictions_within_canoe(arr1, arr2, multipler=multipler)
        val = float(np.sum(c_a_1) / len(c_a_1))
        return val


class SemenAnalysisParameters:
    ca = CanoeAnalysis()

    def __init__(self) -> None:
        self.load_parameters()
        self.get_threshold_data()
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

    def get_threshold_data(self) -> None:
        self.thresholding_info = {}
        for i, row in self.df.iterrows():
            threshold = row["abnormal_threshold"]
            is_lower = row["is_lower"]
            allowable_variance = self.ca.allowable_variance(threshold)
            self.thresholding_info[row["parameter"]] = {"allowable_variance": allowable_variance, "threshold": threshold, "is_lower": is_lower}
        return


class DoughnutAnalysis:
    def get_measurement_status(self, value: float, allowable_variance: float, threshold: int) -> str:
        upper = value + allowable_variance
        lower = value - allowable_variance

        if (upper < threshold) and (value < threshold):
            return 'BELOW TRUE'
        elif (upper >= threshold) and (value < threshold):
            return 'BELOW BOUNDARY'
        elif (lower < threshold):
            return 'ABOVE BOUNDARY'
        return 'ABOVE TRUE'
    
    def get_comparision_status(self, true_val: int | float, pred_val: int | float, allowable_variance: float, threshold: int, is_lower: bool):
        true_status = self.get_measurement_status(true_val, allowable_variance, threshold)
        pred_status = self.get_measurement_status(pred_val, allowable_variance, threshold)

        if is_lower:
            if (true_status in ['BELOW TRUE', 'BELOW BOUNDARY']) and (pred_status in ['BELOW TRUE', 'BELOW BOUNDARY', 'ABOVE BOUNDARY']):
                return "TP"
            elif (true_status in ['BELOW TRUE', 'BELOW BOUNDARY']) and (pred_status == 'ABOVE TRUE'):
                return "FN"
            elif (true_status in ['ABOVE TRUE', 'ABOVE BOUNDARY']) and (pred_status in ['ABOVE TRUE', 'ABOVE BOUNDARY', 'BELOW BOUNDARY']):
                return 'TN'
            elif (true_status in ['ABOVE TRUE', 'ABOVE BOUNDARY']) and (pred_status == 'BELOW TRUE'):
                return 'FP'
        else:
            if (true_status in ['BELOW TRUE', 'BELOW BOUNDARY']) and (pred_status in ['BELOW TRUE', 'BELOW BOUNDARY', 'ABOVE BOUNDARY']):
                return "TN"
            elif (true_status in ['BELOW TRUE', 'BELOW BOUNDARY']) and (pred_status == 'ABOVE TRUE'):
                return "FP"
            elif (true_status in ['ABOVE TRUE', 'ABOVE BOUNDARY']) and (pred_status in ['ABOVE TRUE', 'ABOVE BOUNDARY', 'BELOW BOUNDARY']):
                return 'TP'
            elif (true_status in ['ABOVE TRUE', 'ABOVE BOUNDARY']) and (pred_status == 'BELOW TRUE'):
                return 'FN'
        return None
    
    def get_array_comparision(self, true_vals: np.array, pred_vals: np.array, allowable_variance: float, threshold: int, is_lower: bool):
        res = []
        for tv, pv in zip(true_vals, pred_vals):
            status = self.get_comparision_status(tv, pv, allowable_variance, threshold, is_lower)
            res.append(status)
        return res
    
    def get_boundary_status_array(self, values: np.array, allowable_variance: float, threshold: int) -> str:
        res = []
        for v in values:
            status = self.get_measurement_status(v, allowable_variance, threshold)
            res.append(status)
        return res

    def calculate_doughnut_analysis(self, true_vals: np.array, pred_vals: np.array, allowable_variance: float, threshold: int, is_lower: bool):
        statuses = self.get_array_comparision(true_vals, pred_vals, allowable_variance, threshold, is_lower)
        return {
            "TP": statuses.count("TP"),
            "FP": statuses.count("FP"),
            "TN": statuses.count("TN"),
            "FN": statuses.count("FN"),
        }


class BlandAltmanCalculation:
    CI_95 = 1.96

    def __init__(self):
        return

    def calculate_mean(self, true_values: np.array, predicted_values: np.array) -> np.array:
        return (true_values + predicted_values) / 2

    def calculate_difference(self, true_values: np.array, predicted_values: np.array) -> np.array:
        return true_values - predicted_values

    def calculate_bias(self, measurement_differences: np.array) -> float:
        return np.mean(measurement_differences)

    def calculate_difference_std(self, measurement_differences: np.array) -> float:
        return np.std(measurement_differences)

    def calculate_upper_limit(self, mean_bias: float, difference_std: float) -> float:
        return mean_bias + (self.CI_95 * difference_std)

    def calculate_lower_limit(self, mean_bias: float, difference_std: float) -> float:
        return mean_bias - (self.CI_95 * difference_std)

    def calculate(self, true_values: np.array, predicted_values: np.array) -> dict:
        measurement_means = self.calculate_mean(true_values, predicted_values)
        measurement_differences = self.calculate_difference(true_values, predicted_values)
        mean_bias = self.calculate_bias(measurement_differences)
        difference_std = self.calculate_difference_std(measurement_differences)
        upper_limit = self.calculate_upper_limit(mean_bias, difference_std)
        lower_limit = self.calculate_lower_limit(mean_bias, difference_std)
        return {
            "mean": measurement_means,
            "diff": measurement_differences,
            "mean_bias": mean_bias,
            "diff_std": difference_std,
            "upper_limit": upper_limit,
            "lower_limit": lower_limit,
        }


class PredictionClassification:
    def prediction_status(self, true_abnormal: bool, prediction_abnormal: bool) -> str:
        if true_abnormal and prediction_abnormal:
            return "TP"
        if not true_abnormal and prediction_abnormal:
            return "FP"
        if not true_abnormal and not prediction_abnormal:
            return "TN"
        if true_abnormal and not prediction_abnormal:
            return "FN"
        return None

    def array_prediction_status(self, true_abnormals: List[bool], prediction_abnormals: List[bool]) -> List[str]:
        return [self.prediction_status(t, p) for t, p in zip(true_abnormals, prediction_abnormals)]

    def get_status_counts(self, statuses: List[str]) -> Dict[str, int]:
        return {
            "TP": statuses.count("TP"),
            "FP": statuses.count("FP"),
            "TN": statuses.count("TN"),
            "FN": statuses.count("FN"),
        }

    def calculate_sensitivity(self, status_counts: Dict[str, int]) -> float:
        tp = status_counts["TP"]
        fn = status_counts["FN"]
        return 0 if (tp + fn) == 0 else tp / (tp + fn)

    def calculate_specificity(self, status_counts: Dict[str, int]) -> float:
        tn = status_counts["TN"]
        fp = status_counts["FP"]
        return 0 if (tn + fp) == 0 else tn / (tn + fp)

    def calculate_ppv(self, status_counts: Dict[str, int]) -> float:
        tp = status_counts["TP"]
        fp = status_counts["FP"]
        return 0 if (tp + fp) == 0 else tp / (tp + fp)

    def calculate_npv(self, status_counts: Dict[str, int]) -> float:
        tn = status_counts["TN"]
        fn = status_counts["FN"]
        return 0 if (tn + fn) == 0 else tn / (tn + fn)

    def calculate_accuracy(self, status_counts: Dict[str, int]) -> float:
        tp = status_counts["TP"]
        tn = status_counts["TN"]
        fp = status_counts["FP"]
        fn = status_counts["FN"]
        return (tp + tn) / (tp + tn + fp + fn)

    def calculate_metrics(self, status_counts: Dict[str, int]) -> Dict[str, float]:
        res = {
            "sensitivity": self.calculate_sensitivity(status_counts),
            "specificity": self.calculate_specificity(status_counts),
            "ppv": self.calculate_ppv(status_counts),
            "npv": self.calculate_npv(status_counts),
            "accuracy": self.calculate_accuracy(status_counts),
        }
        res.update(status_counts)
        return res

    def analyse(self, true_abnormals: List[bool], prediction_abnormals: List[bool]) -> Dict[str, float]:
        statuses = self.array_prediction_status(true_abnormals, prediction_abnormals)
        status_counts = self.get_status_counts(statuses)
        return self.calculate_metrics(status_counts)
    
    def analyse_values(self, true_values: np.array, predicted_values: np.array, threshold: float, is_lower: bool) -> Dict[str, float]:
        if is_lower:
            true_abnormals = [True if x <= threshold else False for x in true_values]
            prediction_abnormals = [True if x <= threshold else False for x in predicted_values]
        else:
            true_abnormals = [True if x >= threshold else False for x in true_values]
            prediction_abnormals = [True if x >= threshold else False for x in predicted_values]
        return self.analyse(true_abnormals, prediction_abnormals)


class PredictionComparision:
    bland_altman = BlandAltmanCalculation()
    sap = SemenAnalysisParameters()
    pclass = PredictionClassification()
    doughnut = DoughnutAnalysis()

    def analyse(self, true_values: np.array, predicted_values: np.array, variable: str) -> dict:
        threshold_present = variable in self.sap.thresholding_info

        bland_altman_data = self.bland_altman.calculate(true_values, predicted_values)
        bland_altman_data.pop('mean')
        bland_altman_data.pop('diff')

        res = {}
        res['variable'] = variable
        res.update(bland_altman_data)

        res['c1_score'] = self.sap.ca.calculate_canoe_performance_score(true_values, predicted_values)
        res['c1_5_score'] = self.sap.ca.calculate_canoe_performance_score(true_values, predicted_values, multipler=1.5)

        if threshold_present:
            threshold_info = self.sap.thresholding_info[variable]
            abnormal_threshold = threshold_info['threshold']
            is_lower = threshold_info['is_lower']
            allowable_variance = threshold_info['allowable_variance']
            res['abnormal_threshold'] = abnormal_threshold
            res['is_lower'] = is_lower
            res['allowable_variance_at_threshold'] = allowable_variance

            discrete_performance = self.pclass.analyse_values(true_values, predicted_values, abnormal_threshold, is_lower)
            res.update(discrete_performance)

            doughnut_analysis = self.doughnut.calculate_doughnut_analysis(true_values, predicted_values, allowable_variance, abnormal_threshold, is_lower)
            dn_res = {}
            for k, v in doughnut_analysis.items():
                dn_res[f'DN_{k}'] = v
            res.update(dn_res)
        return res
