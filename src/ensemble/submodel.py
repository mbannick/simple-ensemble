"""
Create a submodel for one covariate-knockout pair.
"""
import matplotlib
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from math import sqrt
import logging
import matplotlib.pyplot as plt
import pylab
from tqdm import tqdm_notebook as tqdm
import rpy2
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from rpy2 import robjects
from rpy2.robjects.packages import importr

from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

nlme = importr('nlme')
lme4 = importr('lme4')
base = importr('base')
stats = importr('stats')

logger = logging.getLogger(__name__)

class Submodel(BaseEstimator, RegressorMixin):
    def __init__(self, response, fixed_effects, random_effects=['super_region', 'region_nest', 'country_nest']):
        """
        Initiates a submodel with the specified response, fixed effects,
        and random effect variables.

        :param response: string of response name used to fit
        :param fixed_effects: list of covariate names
        :param groups: name of variable to use as highest level group random effect
        :param nested: dictionary describing variance components, random effects
                       options for these are region_nest, country_nest, and/or age_nest
        """
        self.response = response
        self.fixed_effects = fixed_effects
        self.random_effects = random_effects
        self.model_formula()

    def model_formula(self):
        """
        Create the string of covariates and outcome to submit to statsmodels.

        :param response: one of ln_rate or lt_cf
        :param covariates: list of covariate names
        :returns: a model formula to use in mixedlm
        """
        self.formula = self.response + " ~"
        for fe in self.fixed_effects:
            self.formula = self.formula + " + " + fe
        for re in self.random_effects:
            self.formula = self.formula + " + (1|" + re + ") "

    def check_convergence(self, model):
        """
        Checks for convergence of the lme4 object.
        """
        messages = model.slots['optinfo'].rx2("conv").rx2("lme4").rx2("messages")
        if tuple(model.rclass)[0] != 'lmerMod':
            return False
        elif type(messages) != rpy2.rinterface.NULLType:
            if "failed to converge" in ' '.join(messages):
                return False
            else:
                return True
        else:
            return True

    def fit(self, df):
        """
        Function fits the regression with given covariates, response,
        and random effects to the given data frame. In model fitting,
        subset to only complete cases.

        :param df: data frame with the response values, fixed effects,
                   and random effect
        """
        self.betas = None
        formula = self.formula

        for re in self.random_effects:
            df[re] = df[re].astype(str)

        df = df.loc[~df[self.response].isnull()]
        robjects.globalenv['dataframe'] = df
        self.mdf = lme4.lmer(formula, data=base.as_symbol('dataframe'))
        
        if not self.check_convergence(self.mdf):
            self.mdf = None
            logger.info("classifier failed to converge")
        else:
            if len(np.array(nlme.fixef(self.mdf))) < len(self.fixed_effects) + 1:
                self.mdf = None
                logger.info("singular matrix -- skipping")
            else:
                self.betas = np.array(nlme.fixef(self.mdf))
        return self

    def predict(self, df, draw=False):
        """
        Predicts y-hat for response variable given a new data frame.
        Will take a draw from the fixed effects if doing draws.

        :param df: data frame with response, fixed and random effects
        :param draw: (bool) want to take draws?
        """
        if self.mdf is None:
            raise NotImplementedError("Oops! This model didn't converge. Try another.")
        betas = np.array(nlme.fixef(self.mdf))
        nm = len(betas)
        if draw:
            vcov = np.array(base.matrix(stats.vcov(self.mdf), nm, nm))
            betas = np.random.multivariate_normal(betas, vcov)
        intercept = np.ones((len(df), 1))
        X = np.hstack([intercept, np.array(df[self.fixed_effects])])
        preds = np.matmul(X, betas)
        ranef = nlme.ranef(self.mdf)
        ranef = dict(zip(ranef.names, list(ranef)))
        res = {}
        for k in ranef.keys():
            RE = ranef[k]
            vals = np.array(base.matrix(RE))
            nams = robjects.StrVector(base.labels(RE)[0])
            nams = [n for n in nams]
            res[k] = dict(zip(nams, vals.ravel().tolist()))
        for re in self.random_effects:
            re_pred = df[re].astype(str).map(res[re]).values
            re_pred = np.nan_to_num(re_pred)
            preds = preds + re_pred
        return preds

    def rmse(self, df):
        """
        Calculates root-mean-squared-error of predictions, y-hat.

        :param df: data frame with response, fixed and random effects
        """
        logger.info("Computing RMSE.")
        if self.mdf is None:
            return np.nan
        else:
            return np.nanmean((self.predict(df) - df[self.response].values)**2)**0.5

    def rmse_mean(self, trainings, tests):
        """
        Calculates average RMSE across training-test sets.
        Don't use this in the ensemble wrapper.
        THIS FUNCTION RE-FITS THE CLASSIFIER

        :param trainings: list of training datasets
        :param tests: list of test datasets
        """
        logger.info("Computing rmse mean across test-training for single classifier.")
        assert len(trainings) == len(tests), "Must pass the same number of test and train datasets!"
        rmse = list()
        for train, test in zip(trainings, tests):
            self.fit(train)
            rmse.append(self.rmse(test))
        return np.nanmean(rmse)

    def generate_draws(self, df, n_draws, refit=True):
        """
        Generate n_draws draws from the submodel on a given
        dataset.

        :param refit: (bool) whether or not to re-fit the
                    classifier on the df passed
        :param n_draws: (int) number of draws to produce
        """
        logger.info("In the generate draws for submodel classifier.")
        if refit:
            logger.info("Refit yes")
            self.fit(df)
        logger.info("Predictions!!")
        logger.info("Going to make n_draws {}".format(n_draws))
        predictions = np.array([self.predict(df, draw=True) for i in range(n_draws)])
        logger.info("Made the preds successfully!")
        return predictions

    def plot_predictions(self, df, querystring, ages, titleadd=""):
        """
        Plot time-series predictions and raw data for a given location
        and a list of ages given.

        :param df: data frame on which to make the predictions
        :param querystring: pandas query to subset df (e.g. by location ID)
        :param ages: list of ages to include
        :param titleadd: (str) string to append onto title
        """
        df['predictions'] = self.predict(df)
        df_sub = df.query(querystring).sort_values('year')
        fig, axes = plt.subplots(1, len(ages), figsize=(20, 10), facecolor='white', sharey='row')
        i = 0
        for axis in axes:
            # sub = df_sub.query('age == {}'.format(ages[i]))
            sub = df_sub.loc[df_sub.age == ages[i]].copy()
            location_name = sub.location.unique()[0]
            axis.plot(sub['year'], sub[self.response], 'ro')
            axis.plot(sub['year'], sub['predictions'])
            axis.set_title(ages[i])
            if self.response == "ln_rate":
                ylabel = "Log Death Rate"
            else:
                ylabel = "Logit Cause Fraction"
            if i == 0:
                axis.set_ylabel(ylabel)
            i+=1
        plt.suptitle('Predictions for {} \n'.format(location_name) + titleadd)

