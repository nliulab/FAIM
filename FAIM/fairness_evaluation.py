import pandas as pd
import numpy as np
from fairlearn.metrics import (
    equalized_odds_difference,
    demographic_parity_difference,
    true_negative_rate,
    selection_rate,
    MetricFrame,
)
from sklearn.metrics import (
    auc,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    recall_score,
    accuracy_score,
    confusion_matrix,
)

# from FairLite.fair_evaluator import FairEvaluator

from .utils import *
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

my_fairness_bases = {
    "tpr": recall_score,
    "tnr": true_negative_rate,
    "sr": selection_rate,
    "acc": accuracy_score,
    "conf_mat": confusion_matrix,
}
# the situation for each group should not be bad; tnr -> fpr
my_bases_bound = {"tpr": 0.6, "tnr": 0.6, "sr": 0, "acc": 0.6, "conf_mat": pd.NA}


def fairarea(fairness_metrics):
    n_metric = len(fairness_metrics)
    tmp = fairness_metrics.values.flatten().tolist()
    tmp_1 = tmp[1:] + tmp[:1]

    if n_metric > 2:
        theta_c = 2 * np.pi / n_metric
        area = np.sum(np.array(tmp) * np.array(tmp_1) * np.sin(theta_c))
    elif n_metric == 2:
        area = np.sum(np.array(tmp) * np.array(tmp_1))
    else:
        area = np.abs(tmp[0])

    return area


class FAIMEvaluator:
    def __init__(
        self,
        y_true,
        y_pred,
        y_pred_bin,
        sen_var,
        fair_only=False,
        cla_metrics=["auc"],
        weighted=False,
        weights=None,
        bases=my_fairness_bases,
        bound=my_bases_bound,
    ):
        """Initialize the fairness evaluator.

        Args:
            y_true: true labels
            y_pred: predicted scores or probabilities
            y_pred_bin: predicted binary labels
            sen_var: the vector of sensitive variables
            fair_only (bool, optional): whether to compute fairness metrics only. Defaults to False.
            cla_metrics (list, optional): classification metrics. Defaults to ["auc"].
            weighted (bool, optional): whether to create a customized fairness metric based on weighted combining of 'tnr' and 'tpr'. Defaults to False.
            weights (_type_, optional): the weights for weighted combining of 'tnr' and 'tpr'. Required when `weighted` is True. Defaults to None.
            bases (_type_, optional): the bases for fairness metrics. Defaults to my_fairness_bases (see above).
            bound (_type_, optional): the bound for base metrics. Defaults to my_bases_bound.
        """
        # super().__init__(y_obs=y_true, y_pred=y_pred, y_pred_bin=y_pred_bin, sens_var=pd.Series(sen_var), y_pos=True)

        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_bin = pd.Series(y_pred_bin)
        self.sen_var = pd.Series(sen_var)
        self.cla_metrics = cla_metrics

        self.my_fairness_bases = bases
        self.my_bases_bound = bound

        if weighted:
            if weights is None or not isinstance(weights, dict):
                raise TypeError(
                    "The weights need to be specified and the type should be dict!"
                )
            elif len(weights) != 2 or np.sum(list(weights.values())) != 1:
                raise ValueError(
                    "The weights should be a dict containing two values respectively for 'tpr' and 'tnr'. In addition, the sum of weights should be equal to 1!"
                )
            else:
                self.weighted = weighted
                self.weights = weights

        self._fairsummary_generation()
        self._fairmetrics_generation()
        if not fair_only:
            if cla_metrics is None:
                raise ValueError("The classification metrics should be specified!")
            self._clametric_generation()

    @staticmethod
    def _check_sen(y_obs, sen_var, sens_var_ref):
        # print("Checking the sensitive variable...")
        return {"sens_var": sen_var, "sens_var_ref": pd.unique(sen_var)[0]}

    def _fairsummary_generation(self):
        """Computation primary metrics (e.g., TPR, TNR, etc.) among subgroups

        Returns:
            _type_: _description_
        """
        self.fairsummary = MetricFrame(
            y_true=self.y_true,
            y_pred=self.y_pred_bin,
            metrics=self.my_fairness_bases,
            sensitive_features=self.sen_var,
        )

    def _fairmetrics_generation(self):
        """Generate 1. machine learning performance-based fairness metrics based on the fairness summary.

        Args:
            fairness_summary (pd.DataFrame): Fairness summary.
            fairness_bases (dict, optional): Fairness bases. Defaults to my_fairness_bases.
            bases_bound (dict, optional): Bases bound. Defaults to my_bases_bound.
            weighted (bool, optional): Whether to create a customized fairness metric based on weighted combining of 'tnr' and 'tpr'. Defaults to False.
            weights (dict, optional): Weights for weighted combining of 'tnr' and 'tpr'. Required when `weighted` is True. Defaults to None.

        Returns:
            tuple: A tuple containing three DataFrames:
                - Fairness metrics DataFrame.
                - QC DataFrame.

        Raises:
            TypeError: If weights are not specified or not of type dict.
            ValueError: If weights do not have two values for 'tpr' and 'tnr', or their sum is not equal to 1.

        """
        # ----- machine learning performance-based fairness metrics -----
        bases = self.my_fairness_bases.keys()
        fairmetrics = {}
        qc = {}
        diff_ = self.fairsummary.difference()
        for b in list(bases)[:-1]:
            qc[b] = self.fairsummary.overall[b] > self.my_bases_bound[b]

        fairmetrics["Equal Opportunity"] = diff_["tpr"]
        fairmetrics["Equalized Odds"] = np.max([diff_["tpr"], diff_["tnr"]])
        fairmetrics["Statistical Parity"] = diff_["sr"]
        fairmetrics["Accuracy Equality"] = diff_["acc"]

        if self.weighted:
            fairmetrics["BER Equality"] = (
                self.weights["tpr"] * diff_["tpr"] + self.weights["tnr"] * diff_["tnr"]
            )

        # ----- calibration-based fairness metrics -----
        # if self.y_pred is None:
        #     warnings.warn("The predicted values by this method is binary only!", category=UserWarning)
        # else:
        #     calib_summary = self.compute_calib()
        #     fairmetrics["Calibration"] = get_cal_fairness(calib_summary)

        self.fairmetrics = pd.DataFrame([fairmetrics])
        self.qc = pd.DataFrame([qc])

    def _clametric_generation(self):
        clametrics = {}
        pred = self.y_pred if self.y_pred is not None else self.y_pred_bin
        if "auc" in self.cla_metrics:
            # clametrics["auc"] = roc_auc_score(self.y_true, self.y_pred)
            clametrics["auc_low"], clametrics["auc"], clametrics["auc_high"] = (
                get_ci_auc(self.y_true, pred, alpha=0.05, type="auc")
            )

        self.clametrics = pd.DataFrame([clametrics])
