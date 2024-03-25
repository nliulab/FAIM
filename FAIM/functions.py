import numpy as np
import pandas as pd
from scipy.stats import t, f
from sklearn.metrics import (
    auc,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def find_y_pred_threshold(target, predicted, method="auc"):
    """Find the optimal probability cutoff point for a classification model related to event rate
    
    Args:
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations

    Returns:
    list type, with optimal cutoff value

    """
    if method == "auc":
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame(
            {
                "tf": pd.Series(tpr - (1 - fpr), index=i),
                "threshold": pd.Series(threshold, index=i),
            }
        )
        thresh = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]

    elif method == "pr-auc":
        precision, recall, threshold = precision_recall_curve(target, predicted)
        i = np.arange(len(precision))
        prc = pd.DataFrame(
            {
                "tf": pd.Series(tpr - (1 - fpr), index=i),
                "threshold": pd.Series(threshold, index=i),
            }
        )
        thresh = prc.iloc[(prc.tf - 0).abs().argsort()[:1]]

    return list(thresh["threshold"])[0]


def compute_ci_cp(numerator, denominator, alpha=0.05):
    """Compute performance metrics with Clopper–Pearson 95\% CI

    Args:
        numerator: Numerator for performance metrics formula.
        denominator: Denominator for performance metrics formula.
        alpha (float, optional): Target error rate. Defaults to 0.05.

    Returns:
        data frame: a data.frame of the performance metrics (\code{est}) and the
        95\% CI (\code{lower}, \code{upper}).
    """
    x = denominator - numerator
    n = denominator

    f_ub = f.ppf(alpha / 2, dfn=2 * x, dfd=2 * (n - x + 1))
    ub = 1 - (1 + (n - x + 1) / (x * f_ub)) ** -1

    f_lb = f.ppf(1 - alpha / 2, dfn=2 * (x + 1), dfd=2 * (n - x))
    lb = 1 - (1 + (n - x) / ((x + 1) * f_lb)) ** -1

    df = pd.DataFrame(
        {"est": numerator / denominator, "lower": lb, "upper": ub}, index=[0]
    )

    return df


def eval_pred(y_pred_bin, y_obs, y_pos=1):
    n = len(y_obs)
    y_neg = list(set(y_obs) - {y_pos})[0]
    TP = np.sum((y_pred_bin == y_pos) & (y_obs == y_pos))
    TN = np.sum((y_pred_bin == y_neg) & (y_obs == y_neg))
    FP = np.sum((y_pred_bin == y_pos) & (y_obs == y_neg))
    FN = np.sum((y_pred_bin == y_neg) & (y_obs == y_pos))
    label_pred = "Predicted +ve"
    df = pd.concat(
        [
            compute_ci_cp(numerator=TP + FP, denominator=n),
            compute_ci_cp(numerator=TP + TN, denominator=n),
            compute_ci_cp(numerator=TP, denominator=TP + FP),
            compute_ci_cp(numerator=TN, denominator=TN + FN),
            compute_ci_cp(numerator=TP, denominator=TP + FN),
            compute_ci_cp(numerator=FP, denominator=FP + TN),
        ]
    )
    df["metric"] = [label_pred, "Accuracy", "PPV", "NPV", "TPR", "FPR"]
    df = df.reset_index(drop=True)
    # Do not report CI for predicted +ve:
    df.loc[df["metric"] == label_pred, ["lower", "upper"]] = [None, None]

    return df


def compute_ci_prop(p, n, alpha=0.05):
    """Compute 95\% CI for a proportion

    Args:
        p: proportion
        n: num of observations
        alpha (float, optional): Target error rate. Default is 0.05 for 95\% CI.

    Returns:
        a vector of the 95% CI.
    """
    se = np.sqrt(p * (1 - p) / (n - 1))
    z = t.ppf(1 - alpha / 2, df=n - 1)
    lower_ci = np.max([p - z * se, 0])
    upper_ci = np.min([p + z * se, 1])
    return [lower_ci, upper_ci]
