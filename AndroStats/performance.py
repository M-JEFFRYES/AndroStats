import pandas as pd
import numpy as np


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
