import pandas as pd
import numpy as np

# import pyarma as arma

import matplotlib.pyplot as plt
import patchworklib as pw

import os
import inspect
from tqdm import tqdm
import time

from ShapleyVIC import model, _util
import shap
import statsmodels.api as sm

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

from .fairness_base import *
from .fairness_evaluation import *
from .fairness_plotting import *
from .utils import *


class FAIMGenerator(FairBase):
    def __init__(
        self,
        dat_train,
        selected_vars,
        selected_vars_cat,
        y_name,
        sen_name,
        sen_var_ref,
        output_dir,
        criterion="loss",
        epsilon=0.05,
        m=800,
        n_final=350,
        without_sen=False,
        pre=False,
        pre_method="Reweigh",
        post=False,
        post_method="equalizedodds",
    ):
        """Initialize the class of FAIM

        Args:
            dat_train (data frame): the training data
            selected_vars (list): the selected variables that include sensitive variables
            selected_vars_cat (list): the selected categorical variables that include sensitive variables
            y_name (str): the name of the label, e.g. "y", "label", etc.
            sen_name (list): the name of the sensitive variables
            sen_var_ref (dict): the reference values of the sensitive variables e.g. {"gender": "F"}
            output_dir: the output directory to store the nearly optimal models results
            criterion (str, optional): the criterion to generate nearly optimal models. Defaults to "loss".
            epsilon (float, optional): the control factor of "nearly optimality", i.e. the gap to the optimal model. Defaults to 0.05.
            without_sen (bool, optional): directly exclude the sensitive variables. Defaults to False.
            pre (bool, optional): whether to use pre-process bias mitigation methods before FAIM. Defaults to False.
            pre_method (str, optional): specific pre-process method. Defaults to "Reweigh".
            post (bool, optional): whether to use post-process bias mitigation methods after FAIM. Defaults to False.
            post_method (str, optional): specific post-process method. Defaults to "EqOdds".
        """

        super().__init__(
            dat_train,
            selected_vars,
            selected_vars_cat,
            y_name,
            sen_name,
            sen_var_ref,
            without_sen,
        )

        self.criterion = criterion
        self.output_dir = output_dir
        self.epsilon = epsilon
        self.m = m
        self.n_final = n_final

        self.pre = pre
        if pre:
            self.pre_method = pre_method
            self.rw_model, self.rw_results, self.rw_weights = self.pre_mitigate()
            plt.hist(self.rw_weights)
        if post:
            self.post = post
            self.post_method = post_method

        self.optim_obj = self.optimal_model(selected_vars, selected_vars_cat)
        self.optim_results = self.optim_obj.model_optim
        self.optim_model = self.optim_obj.model_optim.model

        self.dat_expl = None
        self.dat_test = None

    # def __reduce__(self):
    #     return (self.__class__, (self.coefs, self.best_coef, self.best_optim_base_obj, self.best_sen_exclusion, self.fairmetrics_df))

    def pre_mitigate(self):
        """Pre-process bias mitigation methods"""
        rw_model, rw_results, instance_weights = self.model(
            method_type="pre", method=self.pre_method
        )

        return rw_model, rw_results, instance_weights

    def optimal_model(self, selected_vars, selected_vars_cat):
        """Generate the optimal model

        Args:
            selected_vars (list): the selected variables that can include sensitive variables
            selected_vars_cat (list): the selected categorical variables that can include sensitive variables

        Returns:
            model_object: the object of the optimal model
        """
        x = self.dat_train.drop(
            columns=[
                c
                for c in self.dat_train.columns
                if c == self.y_name or c not in selected_vars
            ]
        )

        model_object = model.models(
            x=x,
            y=self.dat_train[self.y_name],
            x_names_cat=selected_vars_cat,
            output_dir=self.output_dir,
            criterion=self.criterion,
            sample_w=(
                self.rw_weights if self.pre and self.pre_method == "Reweigh" else None
            ) # instance_weights
        )
        return model_object

    def nearly_optimal_model(self, optim_base_obj, m=200, n_final=50, epsilon=None):
        """Generate the nearly optimal models

        Args:
            optim_base_obj (object): the object of the optimal model
            m (int, optional): the number of models to be generated. Defaults to 800.
            n_final (int, optional): the number of nearly optimal models to be generated. Defaults to 350.

        Returns:
            coefs (data frame): the coefficients of the nearly optimal models
            plots (plot): the plot of the status nearly optimal models
        """
        if epsilon is None:
            epsilon = self.epsilon

        u1, u2 = optim_base_obj.init_hyper_params(m=m)
        optim_base_obj.draw_models(
            u1=u1,
            u2=u2,
            m=self.m,
            n_final=self.n_final,
            random_state=1234
        )
        coefs = pd.read_csv(
            os.path.join(self.output_dir, "models_near_optim.csv"), index_col=0
        )
        return coefs, optim_base_obj.models_plot

    def fairness_compute(
        self,
        dat_expl,
        optim_base_obj,
        coefs,
        weighted=True,
        weights={"tnr": 0.5, "tpr": 0.5},
        **kwargs,
    ):
        """Compute the fairness metrics of the nearly optimal models

        Args:
            dat_expl (data frame): the data frame of the validation data
            optim_base_obj (object): the object of the optimal model
            coefs (data frame): the coefficients of the nearly optimal models
            weighted (bool, optional): whether to use weighted fairness metrics. Defaults to True.
            weights (dict, optional): the weights of the weighted fairness metrics. Defaults to {"tnr": 0.5, "tpr": 0.5}.
            **kwargs: the other parameters of the fairness computation

        Returns:
            fairmetrics_df (data frame): the fairness metrics of the nearly optimal models
            qc_df (data frame): the quality control results of the nearly optimal models
        """
        if weighted:
            self.weighted = weighted
            self.weights = weights
        self.dat_expl = dat_expl

        optim_base_results = optim_base_obj.model_optim
        optim_base_model = optim_base_obj.model_optim.model

        fairmetrics_df = []
        qc_df = []
        by_group_list = []

        for i in range(coefs.shape[0]):
            coef = coefs.drop(columns=["perf_metric"]).iloc[i, :]
            x_with_constant, sen_var, y_expl = self.data_process(dat_expl, **kwargs)

            # sen_var = dat_expl[self.sen_name]
            optim_base_results.params = coef

            col_train = optim_base_results.params.index
            col_test = x_with_constant.columns
            x_with_constant = col_gap(col_train, col_test, x_with_constant)

            prob_expl = optim_base_model.predict(params=coef, exog=x_with_constant)
            thresh = find_optimal_cutoff(y_expl, prob_expl)[0]
            pred_expl = prob_expl > thresh

            fe = FAIMEvaluator(
                y_true=y_expl,
                y_pred=prob_expl,
                y_pred_bin=pred_expl,
                sen_var=sen_var,
                fair_only=True,
                weighted=weighted,
                weights=weights,
            )
            fairmetrics = fe.fairmetrics
            qc = fe.qc

            fairmetrics_df.append(fairmetrics)
            qc_df.append(qc)
            by_group_list.append(fe.fairsummary)

        fairmetrics_df = pd.concat(fairmetrics_df).reset_index(drop=True)
        qc_df = pd.concat(qc_df)

        return fairmetrics_df, qc_df
        # self.thresh_list = thresh_list

    def compare(self, dat_expl, optim_base_results, selected_vars, selected_vars_cat):
        """Compare the cases of exclusion of sensitive variables with the original optimal model i.e. no exclusion of sensitive variables

        Args:
            dat_expl (data frame): the data frame of the validation data
            optim_base_results (object): the object of the optimal model regarding the specific case of exclusion of sensitive variables
            selected_vars (list): the selected variables that can include sensitive variables
            selected_vars_cat (list): the selected categorical variables that can include sensitive variables

        Returns:
            bool: whether the case of exclusion of sensitive variables will be expanded to the nearly optimal models

        """
        x_with_constant_ori, sen_var, y_expl = self.data_process(
            dat_expl, selected_vars=self.vars, selected_vars_cat=self.vars_cat
        )
        x_with_constant_base, sen_var, y_expl = self.data_process(
            dat_expl, selected_vars=selected_vars, selected_vars_cat=selected_vars_cat
        )
        pred_ori = self.optim_results.predict(x_with_constant_ori)
        pred_base = optim_base_results.predict(x_with_constant_base)

        if self.criterion == "auc":
            auc_ori = roc_auc_score(y_expl, pred_ori)
            auc_base = roc_auc_score(y_expl, pred_base)

            # return auc_base > auc_ori * (1-np.sqrt(self.epsilon))
            return auc_base > auc_ori * np.sqrt(1 - self.epsilon)
        if self.criterion == "loss":
            loss_ori = self.optim_model.loglike(self.optim_results.params)
            loss_base = optim_base_results.model.loglike(optim_base_results.params)
            ratio = loss_base / loss_ori
            print(f"loss_ori: {loss_ori}, loss_base: {loss_base}, ratio: {ratio}")
            return ratio < np.sqrt(1 + self.epsilon)

    def FAIM_model(self, dat_expl):
        """FAIM: Generate the nearly optimal models and compute the fairness metrics of the nearly optimal models

        Args:
            dat_expl (data frame): the data frame of the validation data
        """
        self.dat_expl = dat_expl

        self.coefs = {}
        self.plots = {}
        self.optim_base_obj_list = {}
        self.fairmetrics_df = pd.DataFrame()
        self.qc_df = pd.DataFrame()

        if self.without_sen == "auto":
            sen_senarios = generate_subsets(self.sen_name)
            pbar = tqdm(
                sen_senarios, desc="Generating nearly optimal models", postfix="*Start*"
            )
            for x in pbar:
                pbar.set_postfix(postfix=f"exclusion: {x}")

                selected_vars = [i for i in self.vars if i not in x]
                selected_vars_cat = [i for i in self.vars_cat if i not in x]
                optim_base_obj = self.optimal_model(selected_vars, selected_vars_cat)
                optim_base_results = optim_base_obj.model_optim

                if self.compare(
                    dat_expl, optim_base_results, selected_vars, selected_vars_cat
                ):
                    self.optim_base_obj_list["_".join(x)] = optim_base_obj

                    if self.criterion == "auc":
                        epsilon = 1 - np.sqrt(1 - self.epsilon)
                    elif self.criterion == "loss":
                        epsilon = np.sqrt(1 + self.epsilon) - 1

                    coefs, plots = self.nearly_optimal_model(
                        optim_base_obj, n_final=self.n_final, epsilon=epsilon
                    )
                    self.coefs["_".join(x)] = coefs
                    self.plots["_".join(x)] = plots
                    fairmetrics_df, qc_df = self.fairness_compute(
                        dat_expl,
                        optim_base_obj,
                        coefs,
                        selected_vars=selected_vars,
                        selected_vars_cat=selected_vars_cat,
                    )
                    fairmetrics_df["auc"] = coefs["perf_metric"]
                    qc_df["auc"] = coefs["perf_metric"]

                    fairmetrics_df["sen_var_exclusion"] = "_".join(x)
                    qc_df["sen_var_exclusion"] = "_".join(x)
                    self.fairmetrics_df = pd.concat(
                        [self.fairmetrics_df, fairmetrics_df]
                    )
                    self.qc_df = pd.concat([self.qc_df, qc_df])

                else:
                    print(f"Exclusion of {x} degrades the discrimination performance!")

            self.fairmetrics_df = self.fairmetrics_df.reset_index(drop=True)
            self.qc_df = self.qc_df.reset_index(drop=True)

            # return fairmetrics_df, qc_df

    def describe(self, selected_metrics=None):
        """Describe the distribution of fairness metrics for all nearly optimal models
        Args:
            selected_metrics (list, optional): the selected fairness metrics, e.g. ["Statistical Parity", "Equalized Odds", "Average Accuracy"]. Defaults to None.

        Returns:
            fig: the plot of the distribution of fairness metrics for all nearly optimal models
        """
        sen_var_exclusion = self.fairmetrics_df["sen_var_exclusion"]
        auc_var = self.fairmetrics_df["auc"]
        fairmetrics_df = self.fairmetrics_df.drop(columns=["sen_var_exclusion"])
        qc_df = self.qc_df.drop(columns=["sen_var_exclusion", "auc"])

        if selected_metrics is None:
            num_metrics = fairmetrics_df.shape[1]
        else:
            for m in selected_metrics:
                if m not in fairmetrics_df.columns:
                    raise ValueError(f"The metric {m} is not in the fairness metrics!")

            qc_df = qc_df[selected_metrics]
            fairmetrics_df = fairmetrics_df[selected_metrics]
            num_metrics = len(selected_metrics)

        ids_after_qc = np.arange(qc_df.shape[0])[
            (np.sum(qc_df, 1) == qc_df.shape[1]).tolist()
        ]
        print(f"{len(ids_after_qc)} are qualified after quality control")

        min_ones = fairmetrics_df.iloc[ids_after_qc, :].apply(axis=0, func=np.argmin)

        for m in min_ones.index:
            id = fairmetrics_df.iloc[ids_after_qc, :].index[min_ones[m]]
            print(
                f"the model with minimal {m}: No.{id} -- {fairmetrics_df.loc[id, m]:.3f}, with {sen_var_exclusion[id]} excluded from regression"
            )

        plot_df = self.fairmetrics_df.loc[ids_after_qc, :]
        fig = plot_distribution(plot_df)

        if len(ids_after_qc) < fairmetrics_df.shape[0]:
            return (
                fairmetrics_df.iloc[ids_after_qc, :],
                [sen_var_exclusion[i] for i in ids_after_qc],
                fig,
            )

        return fig

    def transmit(
        self,
        targeted_metrics=["Average Accuracy", "Statistical Parity", "Equalized Odds"],
        thresh_show=0.3,
        best_id=None,
        best_sen_exclusion=None,
        **kwargs,
    ):
        """Select the best model regarding fairness

        Args:
            targeted_metrics (list, optional): the targeted fairness metrics. Defaults to ["Average Accuracy", "Statistical Parity", "Equalized Odds"].
            thresh_show (float, optional): the threshold to filter the models. Defaults to 0.3.

        Returns:
            best_coef: the coefficients of the best model
            best_sen_exclusion: the sensitive variables excluded from the best model
            best_optim_base_obj: the object of the best model
            p: the radar plot of the distribution of fairness metrics for all nearly optimal models
        """
        FAIM_area_list = []
        ids = np.sum(
            self.fairmetrics_df[targeted_metrics] < thresh_show, axis=1
        ) == len(targeted_metrics)
        if len(ids) == 0:
            raise ValueError("The thresh is too low!")
        fairmetrics_df = self.fairmetrics_df.loc[ids, :]

        sen_var_exclusion = fairmetrics_df["sen_var_exclusion"]
        perf = fairmetrics_df["auc"]
        df = fairmetrics_df.drop(columns=["sen_var_exclusion"])
        df = df[targeted_metrics]

        print(f"There are {df.shape[0]} models for final fairness selection.")

        if "title" in kwargs.keys():
            title = kwargs["title"]
        else:
            title = None
        p_radar = plot_radar(df, thresh_show=thresh_show, title=title)
        # p_radar.show()
        p, fair_idx_df = plot_scatter(df, perf, sen_var_exclusion, title=title)
        self.p = p
        p.show()

        for i, id in enumerate(df.index):
            values = df.loc[id, :]
            FAIM_area_list.append(fairarea(values))
        ranking = np.argsort(np.argsort(FAIM_area_list))
        
        if best_id is not None:
            assert best_sen_exclusion is not None
            self.best_id = best_id
            self.best_sen_exclusion = best_sen_exclusion
        else:
            self.best_id = df.index[np.where(ranking == 0)][0]
            self.best_sen_exclusion = sen_var_exclusion.iloc[np.argmin(FAIM_area_list)]

        id_senario = [
            index
            for index, item in enumerate(list(self.coefs.keys()))
            if item == self.best_sen_exclusion
        ][0]

        self.best_coef = (
            self.coefs[self.best_sen_exclusion]
            .drop(columns=["perf_metric"])
            .loc[self.best_id - self.n_final * id_senario, :]
        )
        self.best_optim_base_obj = self.optim_base_obj_list[self.best_sen_exclusion]

        # confidence interval
        dat_uncertainty = self.dat_train.sample(
            n=np.min([50000, self.dat_train.shape[0]]), random_state=42
        )
        excluded_vars = self.best_sen_exclusion.split("_")
        selected_vars = [i for i in self.vars if i not in excluded_vars]
        selected_vars_cat = [i for i in self.vars_cat if i not in excluded_vars]
        x_with_constant = self.data_process(
            dat_uncertainty,
            selected_vars=selected_vars,
            selected_vars_cat=selected_vars_cat,
        )[0].values

        prob_train, _, _ = self.test(dat_uncertainty)
        best_se = None
        fisher_information = (
            x_with_constant.T @ np.diag(prob_train * (1 - prob_train)) @ x_with_constant
        )
        print("multiplication successed!")
        cov = np.linalg.pinv(fisher_information)
        best_se = [np.sqrt(cov[i, i]) for i in range(cov.shape[0])]

        # self.best_thresh = self.thresh_list[self.best_id]
        best_results = {
            "best_coef": self.best_coef,
            "best_sen_exclusion": self.best_sen_exclusion,
            "best_se": best_se,
            "best_optim_base_obj": self.best_optim_base_obj,
        }

        return best_results, fair_idx_df

    def post_mitigate(self):
        pass

    def test(self, dat_test, model=None, params=None, thresh=None):
        """Test the best model regarding fairness

        Args:
            dat_test (data frame): the data frame of the test data
            model (object, optional): the object of the model to be tested. Defaults to None.
            params (optional): the parameters of the model to be tested. Defaults to None.
            thresh (optional): the threshold of the predictions. Defaults to None.
        
        Methods:
        +--------------+------------+----------------------------------------+
        | Model        | Params     | Description                            |
        +--------------+------------+----------------------------------------+
        | None         | None       | Test the best model produced by FAIM.  |
        +--------------+------------+----------------------------------------+
        | model results| None       | Test the provided model with parameters|
        |              |            | embedded.                              |
        +--------------+------------+----------------------------------------+
        | model results| as required| Test the provided model with the       |
        |              |            | parameters additionally provided.      |
        +--------------+------------+----------------------------------------+
        
        Returns:
            prob_test: the predicted probabilities of the test data
            fairmetrics: the fairness metrics of the test data
            fairsummary: the fairness summary of the test data for each subgroup
                    
        """
        self.dat_test = dat_test
        excluded_vars = self.best_sen_exclusion.split("_")
        selected_vars = [i for i in self.vars if i not in excluded_vars]
        selected_vars_cat = [i for i in self.vars_cat if i not in excluded_vars]
        x_with_constant, sen_var, y_test = self.data_process(
            dat_test, selected_vars=selected_vars, selected_vars_cat=selected_vars_cat
        )

        if model is None:
            prob_test = self.best_optim_base_obj.model_optim.model.predict(
                params=self.best_coef, exog=x_with_constant
            )
        else:
            if isinstance(model, type(self.optim_results)) and params is None:
                prob_test = model.predict(exog=x_with_constant)
            elif isinstance(model, type(self.optim_model)) and params is not None:
                prob_test = model.predict(params=params, exog=x_with_constant)
            else:
                raise ValueError("Please provide the right model!")

        thresh = find_optimal_cutoff(y_test, prob_test)[0]
        pred_test = prob_test > thresh
        fe = FAIMEvaluator(
            y_true=np.array(y_test),
            y_pred_bin=pred_test,
            y_pred=prob_test,
            sen_var=sen_var,
            weighted=self.weighted,
            weights=self.weights,
        )
        fairmetrics = fe.fairmetrics
        clametrics = fe.clametrics

        return prob_test, fairmetrics, clametrics

    def explain(self, method="best"):
        """compute the SHAP values of the best model and original model

        Args:
            method (str, optional): "best" or "ori". Defaults to "best".
        """

        def f(bg):
            model = self.best_optim_base_obj.model_optim.model
            if method == "best":
                return model.predict(params=self.best_coef, exog=bg)
            else:
                return model.predict(params=self.optim_results.params, exog=bg)

        output_dir = os.path.join(self.output_dir, "explain")
        if method == "best":
            excluded_vars = self.best_sen_exclusion.split("_")
            selected_vars = [i for i in self.vars if i not in excluded_vars]
            selected_vars_cat = [i for i in self.vars_cat if i not in excluded_vars]

        else:
            selected_vars = self.vars
            selected_vars_cat = self.vars_cat

        bg_data = self.data_process(
            self.dat_train,
            selected_vars=selected_vars,
            selected_vars_cat=selected_vars_cat,
        )[0].sample(n=1000, random_state=42)
        ex_data = self.data_process(
            self.dat_expl,
            selected_vars=selected_vars,
            selected_vars_cat=selected_vars_cat,
        )[0].sample(n=200, random_state=42)

        e = shap.KernelExplainer(f, bg_data)
        shap_values_train = e.shap_values(ex_data)
        shap_values_train_1 = shap_values_train[1].squeeze()
        pd.DataFrame(shap_values_train_1).to_csv(
            os.path.join(output_dir, f"{method}.csv")
        )

        return shap_values_train_1

    def compare_explain(self, overide=True):
        """Compare the SHAP values of the best model and original model

        Args:
            overide (bool, optional): Can save time and reproduce the figure if the shap values have been calculated and saved. Defaults to True.
        """
        output_dir = os.path.join(self.output_dir, "explain")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if (not os.path.exists(os.path.join(output_dir, "best.csv"))) or overide:
            shap_best = self.explain(method="best")
        else:
            shap_best = pd.read_csv(
                os.path.join(output_dir, "best.csv"), index_col=0
            ).values.reshape(-1)

        if (not os.path.exists(os.path.join(output_dir, "ori.csv"))) or overide:
            shap_ori = self.explain(method="ori")
        else:
            shap_ori = pd.read_csv(
                os.path.join(output_dir, "ori.csv"), index_col=0
            ).values.reshape(-1)

        p_best = plot_bar(
            shap_best,
            feature_names=self.best_coef.index,
            original_feature_names=self.vars,
            title="Fairness-aware model (FAIM)",
            color="steelblue" #"#D4AF37"
        )
        p_ori = plot_bar(
            shap_ori,
            feature_names=self.optim_results.params.index,
            original_feature_names=self.vars,
            title="Fairness-unaware model (Baseline)",
            color="orange" #"grey"
        )

        f1 = pw.load_ggplot(p_best, figsize=(5, 5))
        f2 = pw.load_ggplot(p_ori, figsize=(5, 5))
        f12 = f2 | f1

        return f12
