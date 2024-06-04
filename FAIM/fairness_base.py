import copy
import statsmodels.api as sm
from ShapleyVIC import model, _util
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
from aif360.algorithms.postprocessing.calibrated_eq_odds_postprocessing import (
    CalibratedEqOddsPostprocessing,
)
from aif360.algorithms.postprocessing.reject_option_classification import (
    RejectOptionClassification,
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from sklearn.linear_model import LogisticRegression

from .utils import *
from .fairness_evaluation import *
import matplotlib.pyplot as plt

seed = 1234
np.random.seed(seed)
rng = np.random.RandomState(seed)


class FairBase:
    def __init__(
        self,
        dat_train,
        selected_vars,
        selected_vars_cat,
        y_name,
        sen_name,
        sen_var_ref,
        without_sen=False,
        weighted=True,
        weights={"tnr": 0.5, "tpr": 0.5},
    ):
        """Initialize the fairness base class

        Args:
            dat_train (data frame): training data
            selected_vars (list): selected variables including sensitive variables
            selected_vars_cat (list): selected categorical variables
            y_name (str): the name of the label
            sen_name (list): the name of the sensitive variable
            sen_var_ref (dict): the reference level of the sensitive variables
            without_sen (bool, optional): directly exclude the sensitive variables. Defaults to False.
            weighted (bool, optional): compute the weighted version of metrics "tnr" and "tpr". Defaults to True.
            weights (dict, optional): the weightage for "tnr" and "tpr", summing up to 1. Defaults to {"tnr": 0.5, "tpr": 0.5}.
        """

        self.dat_train = dat_train
        self.vars = selected_vars
        self.vars_cat = selected_vars_cat
        self.y_name = y_name

        if not isinstance(sen_name, list):
            self.sen_name = [self.sen_name]
        for s in sen_name:
            if sen_var_ref[s] not in dat_train[s].unique():
                raise ValueError(
                    f"Please provide the right reference level of sensitive variables {s}!"
                )
            if s not in self.vars:
                self.vars.append(s)
        else:
            self.sen_name = sen_name
            self.sen_var_ref = sen_var_ref

        self.without_sen = without_sen
        self.weighted = weighted
        self.weights = weights

    def data_process(
        self, dat, selected_vars=None, selected_vars_cat=None, without_sen=None
    ):
        """Data preprocess

        Args:
            dat (data frame): data
            selected_vars (list, optional): selected variables (can include sensitive variables). This needs to be provided if the case considered is beyond completely inclusion or exclusion of sensitive variables. Defaults to None.
            selected_vars_cat (list, optional): selected categorical variables, subset of selected variables. Defaults to None.
            without_sen (bool, optional): directly exclude the sensitive variables. Defaults to None.

        Returns:
            x_1: predictors with one-coding and with constant
            sen_var: the vector of sensitive variable. combined by "_", if there are several sensitive variables
            y: the vector of the label

        """

        def combine_sen(dat, sen):
            new_sen = ["_".join(v) for v in zip(*[dat[s].astype("str") for s in sen])]
            return new_sen

        if selected_vars is None:
            selected_vars = self.vars
            selected_vars_cat = self.vars_cat

        x = dat.drop(
            columns=[
                c for c in dat.columns if c == self.y_name or c not in selected_vars
            ]
        )

        if (
            self.without_sen != "auto"
            and (without_sen is None and self.without_sen)
            or without_sen
        ):
            x = x.drop(columns=self.sen_name)

        if len(self.sen_name) > 1:
            sen_var = combine_sen(dat, self.sen_name)
        else:
            sen_var = dat[self.sen_name[0]]

        y = dat[self.y_name]

        x_dm, x_groups = _util.model_matrix(x=x, x_names_cat=selected_vars_cat)
        x_1 = sm.add_constant(x_dm).astype("float")
        return x_1, sen_var, y

    def data_prepare(self, dat_expl=None):
        """Shape the data to AIF360 format

        Args:
            dat_expl (_type_, optional): validation data needed for post-processing methods. Defaults to None.

        """
        if self.method == "Unawareness":
            x_with_constant, _, y_train = self.data_process(
                self.dat_train if dat_expl is None else dat_expl, without_sen=True
            )
            return x_with_constant

        if self.method_type == "pre":
            x_with_constant, _, y_train = self.data_process(self.dat_train)
            x_with_constant_sen_bin = copy.deepcopy(x_with_constant)
            for s in self.sen_name:
                x_with_constant_sen_bin[s] = [
                    0 if i == self.sen_var_ref[s] else 1 for i in self.dat_train[s]
                ]
            # print(x_with_constant_expl_sen_bin.columns.head(), flush=True)

            pre_train_df = pd.concat([x_with_constant_sen_bin, y_train], axis=1)
            pre_train = BinaryLabelDataset(
                favorable_label=1,
                df=pre_train_df,
                label_names=["label"],
                protected_attribute_names=self.sen_name,
            )
            return pre_train

        elif self.method_type == "post":
            if dat_expl is None:
                raise ValueError("Please provide validation data.")
            else:
                x_with_constant_expl, sen_var, y_expl = self.data_process(dat_expl)
                x_with_constant_expl_sen_bin = copy.deepcopy(x_with_constant_expl)

                for s in self.sen_name:
                    x_with_constant_expl_sen_bin[s] = [
                        0 if i == self.sen_var_ref[s] else 1 for i in dat_expl[s]
                    ]

                prob_expl_ori = self.lr_results.predict(x_with_constant_expl)
                ori_thresh = find_optimal_cutoff(y_expl, prob_expl_ori)[0]
                pred_expl_ori = prob_expl_ori > ori_thresh

                post_expl_df = pd.concat([x_with_constant_expl_sen_bin, y_expl], axis=1)
                post_expl = BinaryLabelDataset(
                    favorable_label=1,
                    df=post_expl_df,
                    label_names=["label"],
                    protected_attribute_names=self.sen_name,
                )
                post_expl_pred = post_expl.copy(deepcopy=True)
                post_expl_pred.scores = prob_expl_ori.values.reshape(-1, 1)
                post_expl_pred.labels = pred_expl_ori.values.reshape(-1, 1)
                return post_expl, post_expl_pred

    def model(self, method_type=None, method=None, dat_expl=None, **kwargs):
        """Fit the model

        Args:
            method_type (str, optional): the type of bias mitigation method (pre/in/post). Defaults to None.
            method (str, optional): the name of bias mitigation method. Defaults to None.
            dat_expl (_type_, optional): validation data needed for post-processing methods. Defaults to None.

            Methods:
            +------------------+--------------------------------+
            | Method type      | Specific methods               |
            +==================+================================+
            | None             | "OriginalLR", "Unawareness"   |
            +------------------+--------------------------------+
            | "pre"            | "Reweigh"                      |
            +------------------+--------------------------------+
            | "in"             | "Reductions"                   |
            +------------------+--------------------------------+
            | "post"           | "EqOdds", "CalEqOdds", "ROC"  |
            +------------------+--------------------------------+

        Returns:
            model results that can be used for prediction
        """
        self.method = method
        self.method_type = method_type

        if isinstance(self.sen_name, list):
            privileged_groups = [{s: 0 for s in self.sen_name}]
            unprivileged_groups = [{s: 1 for s in self.sen_name}]
        else:
            privileged_groups = [{self.sen_name: 0}]
            unprivileged_groups = [{self.sen_name: 1}]

        x_with_constant_nosen, _, y_train = self.data_process(
            self.dat_train, without_sen=True
        )
        x_with_constant, _, y_train = self.data_process(self.dat_train)

        # original LR
        lr_model = sm.GLM(
            self.dat_train[self.y_name], x_with_constant, family=sm.families.Binomial()
        )
        self.lr_results = lr_model.fit()

        if method_type == None:
            if method == "OriginalLR":
                return self.lr_results

            elif method == "Unawareness":
                un_model = sm.GLM(
                    self.dat_train[self.y_name],
                    x_with_constant_nosen,
                    family=sm.families.Binomial(),
                )
                un_results = un_model.fit()
                return un_results
            else:
                raise ValueError(
                    "Please confirm the method: 'OriginalLR' if no bias mitigation is needed; 'Unawareness' if simply excluding the sensitive variabl is enough."
                )

        elif method_type == "pre":
            pre_train = self.data_prepare()

            if method == "Reweigh":
                reweigh_model = Reweighing(
                    privileged_groups=privileged_groups,
                    unprivileged_groups=unprivileged_groups,
                )
                rw_train = reweigh_model.fit_transform(pre_train)
                rw_model = sm.GLM(
                    self.dat_train[self.y_name],
                    x_with_constant,
                    family=sm.families.Binomial(),
                    freq_weights=rw_train.instance_weights,
                )
                plt.hist(rw_train.instance_weights)
                rw_results = rw_model.fit()
                return rw_model, rw_results, rw_train.instance_weights
            else:
                raise ValueError(
                    "Please specify the type of pre-process bias mitigation method among ['Reweigh']!"
                )

        elif method_type == "in":
            if method == "Reductions":

                constraint = EqualizedOdds(difference_bound=0.01)
                np.random.seed(
                    0
                )  # set seed for consistent results with ExponentiatedGradient
                lr_model_sk = LogisticRegression(max_iter=5000, penalty=None)
                mitigator = ExponentiatedGradient(lr_model_sk, constraint)

                mitigator.fit(
                    x_with_constant,
                    y_train,
                    sensitive_features=self.dat_train[self.sen_name],
                )
                return mitigator
            else:
                raise ValueError(
                    "Please specify the type of in-process bias mitigation method among ['Reductions']!"
                )

        elif method_type == "post":
            post_expl, post_expl_pred = self.data_prepare(dat_expl=dat_expl)

            if method == "EqOdds":
                eq_model = EqOddsPostprocessing(
                    privileged_groups=privileged_groups,
                    unprivileged_groups=unprivileged_groups,
                    seed=seed,
                )
                eq_results = eq_model.fit(post_expl, post_expl_pred)
                return eq_results

            elif method == "CalEqOdds":
                if "cost_constraint" in kwargs:
                    cost_constraint = kwargs["cost_constraint"]
                else:
                    cost_constraint = "weighted"  # "fnr", "fpr", "weighted"
                cal_eq_model = CalibratedEqOddsPostprocessing(
                    privileged_groups=privileged_groups,
                    unprivileged_groups=unprivileged_groups,
                    cost_constraint=cost_constraint,
                    seed=seed,
                )
                cal_eq_results = cal_eq_model.fit(post_expl, post_expl_pred)
                return cal_eq_results

            elif method == "ROC":
                ub = 0.05 if "ub" not in kwargs else kwargs["ub"]
                lb = -0.05 if "lb" not in kwargs else kwargs["lb"]
                metric_name = (
                    "Equal opportunity difference"
                    if "metric_name" not in kwargs
                    else kwargs["metric_name"]
                )
                # allowed_metrics = ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"]
                ROC_model = RejectOptionClassification(
                    privileged_groups=privileged_groups,
                    unprivileged_groups=unprivileged_groups,
                    low_class_thresh=0.01,
                    high_class_thresh=0.99,
                    num_class_thresh=100,
                    num_ROC_margin=50,
                    metric_name=metric_name,
                    metric_ub=ub,
                    metric_lb=lb,
                )
                ROC_results = ROC_model.fit(post_expl, post_expl_pred)
                return ROC_results

            else:
                raise ValueError(
                    "Please specify the type of post-process bias mitigation method among ['EqOdds', 'CalEqOdds', 'ROC']!"
                )

        else:
            raise ValueError(
                "Please specify the type of bias mitigation method (pre/in/post)!"
            )

    def test(self, dat_test, model=None, params=None, thresh=None, **kwargs):
        """Test the fairness of the model

        Args:
            dat_test (data frame): test data
            model (_type_, optional): the fitted model. Defaults to None.
            params (_type_, optional): coefficients for the model. Defaults to None.
            thresh (_type_, optional): threshold for the predictions. Defaults to None.

        Returns:
            pred_test / prob_test (array): predicted labels / predicted probabilities
            fairmetrics (data frame): fairness metrics
            fairsummary (data frame): fairness summary for each subgroup
        """
        if "without_sen" in kwargs.keys():
            without_sen = kwargs["without_sen"]
        else:
            without_sen = self.without_sen
        x_with_constant_test, sen_var, y_test = self.data_process(dat_test)
        prob_test = None
        thresh = None

        if model is not None:
            if self.method_type == "post":
                _, post_test_pred = self.data_prepare(dat_expl=dat_test)
                pred_test = model.predict(post_test_pred).labels.reshape(-1)
            else:
                if self.method_type == None and self.method == "Unawareness":
                    x_with_constant_test = self.data_prepare(dat_expl=dat_test)

                if self.method == "Reductions":
                    pred_test = model.predict(x_with_constant_test)
                else:
                    prob_test = model.predict(x_with_constant_test)
                    thresh = find_optimal_cutoff(y_test, prob_test)[0]
                    pred_test = prob_test > thresh
                    # print(prob_test)
        else:
            raise ValueError("Please provide the right model!")

        fe = FAIMEvaluator(
            y_true=y_test,
            y_pred=prob_test,
            y_pred_bin=pred_test,
            sen_var=sen_var,
            weighted=self.weighted,
            weights=self.weights,
        )
        fairmetrics = fe.fairmetrics
        fairsummary = fe.fairsummary
        clametrics = fe.clametrics

        return pred_test if prob_test is None else prob_test, fairmetrics, clametrics
