from typing import List, Dict


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

    def array_prediction_status(
        self, true_abnormals: List[bool], prediction_abnormals: List[bool]
    ) -> List[str]:
        return [
            self.prediction_status(t, p)
            for t, p in zip(true_abnormals, prediction_abnormals)
        ]

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
