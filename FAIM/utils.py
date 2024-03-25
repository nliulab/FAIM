import pandas as pd
import numpy as np
from sklearn.metrics import (
    auc,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    recall_score,
    accuracy_score,
    confusion_matrix,
)
import itertools
from collections import Counter
from itertools import combinations
from scipy.interpolate import CubicSpline
from scipy import integrate
from PIL import Image, ImageDraw, ImageFont

seed = 1234
np.random.seed(seed)
rng = np.random.RandomState(seed)


# metrics
def get_ci_auc(y_true, y_pred, alpha=0.05, type="auc"):
    """Calculate the confidence interval for the AUC (Area Under the Curve) score
    or PR (Precision-Recall) score using bootstrapping.

    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted scores or probabilities.
        alpha (float, optional): Significance level for the confidence interval. Default is 0.05.
        type (str, optional): Type of score to calculate: 'auc' (default) or 'pr' (precision-recall).

    Returns:
        tuple: Tuple containing the lower and upper bounds of the confidence interval.
    """

    n_bootstraps = 1000
    bootstrapped_scores = []

    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred) - 1, len(y_pred))

        if len(np.unique(y_true[indices])) < 2:
            continue

        if type == "pr":
            precision, recall, thresholds = precision_recall_curve(
                y_true[indices], y_pred[indices]
            )
            score = auc(recall, precision)
        else:
            score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()

    # 95% c.i.
    confidence_lower = sorted_scores[int(alpha / 2 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(1 - alpha / 2 * len(sorted_scores))]

    return confidence_lower, np.median(sorted_scores), confidence_upper


def find_optimal_cutoff(target, predicted, method="auc"):
    """Find the optimal probability cutoff point for a classification model related to event rate.

    Args:
        target (array-like): True labels.
        predicted (array-like): Predicted scores or probabilities.
        method (str, optional): Method for finding the optimal cutoff. Default is 'auc'.

    Returns:
        list: List of optimal cutoff values.
    """
    if method == "auc":
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame(
            {
                "tf": pd.Series(tpr + (1 - fpr), index=i),
                "threshold": pd.Series(threshold, index=i),
            }
        )
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[::-1][:1]]
    elif method == "pr-auc":
        precision, recall, threshold = precision_recall_curve(target, predicted)
        i = np.arange(len(precision))
        prc = pd.DataFrame(
            {
                "tf": pd.Series(tpr - (1 - fpr), index=i),
                "threshold": pd.Series(threshold, index=i),
            }
        )
        prc_t = prc.iloc[(prc.tf - 0).abs().argsort()[:1]]

    return list(roc_t["threshold"])


def get_cal_fairness(df):
    def absolute_difference(x):
        return np.abs(spline1(x) - spline0(x))

    df.groupby("group").apply(lambda x: np.max(x["p_obs"]))
    # for g in df_calib.group.unique():

    gs = df.group.unique()
    pairs = list(combinations(gs, 2))

    x_min_thresh = np.min(df.groupby("group").apply(lambda x: np.min(x["p_pred"])))
    x_max_thresh = np.max(df.groupby("group").apply(lambda x: np.max(x["p_pred"])))
    num_points = 100
    diff_cal = []

    for p in pairs:
        p0 = df.loc[df.group == p[0], ["p_obs", "p_pred"]].sort_values(
            by="p_pred", ascending=True
        )
        p1 = df.loc[df.group == p[1], ["p_obs", "p_pred"]].sort_values(
            by="p_pred", ascending=True
        )

        x0 = p0["p_pred"]
        y0 = p0["p_obs"]
        spline0 = CubicSpline(x0, y0)

        x1 = p1["p_pred"]
        y1 = p1["p_obs"]
        spline1 = CubicSpline(x1, y1)

        x_sample = np.linspace(x_min_thresh, x_max_thresh, num_points)
        y_sample = absolute_difference(x_sample)
        area = integrate.simpson(y_sample, x_sample)
        diff_cal.append(area)

    cal_metric = np.mean(diff_cal)
    # print(f"Calibration metric: {cal_metric:.2f}")
    return cal_metric


## small functions
def col_gap(col_train, col_test, x_with_constant):
    if len(col_train) != len(col_test):
        col_gap = [i not in col_test for i in col_train]
        x_with_constant[col_train[col_gap]] = 0
        x_with_constant = x_with_constant.loc[:, col_train]

    return x_with_constant


def generate_subsets(input_list):
    subsets = []
    n = len(input_list)
    for subset_size in range(n + 1):
        for subset in itertools.combinations(input_list, subset_size):
            subsets.append(list(subset))
    return subsets
