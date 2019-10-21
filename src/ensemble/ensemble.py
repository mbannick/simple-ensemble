import matplotlib
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import pylab
import logging

from tqdm import tqdm as tqdm

logger = logging.getLogger(__name__)

class Ensemble(object):
    def __init__(self, classifiers=None):
        if classifiers is None:
            self.classifiers = []
        else:
            self.classifiers = classifiers
        self.weights = None
        self.ranks = None
        self.classifier_params = list()

    def add(self, classifier):
        self.classifiers.append(classifier)

    def get_classifier_params(self):
        """
        Retrieve a list of all of the parameters in each of the classifiers.
        """
        for clf in self.classifiers:
            self.classifier_params.append(clf.get_params())
        return self.classifier_params

    def fit(self, df):
        """
        Fit each of the classifiers
        to the given dataset.

        Each time fit is called, it updates the model fit
        so if you want to fit AND predict to new data,
        you have to fit() and predict().
        """
        i = 0
        for clf in self.classifiers:
            logger.info("Fitting classifier {}".format(i))
            clf.fit(df)
            i+=1
        return self

    def classifier_rmse(self, df):
        """
        Calculates RMSE of predictions y-hat for each data frame passed.
        """
        logger.info("Computing classifier RMSE")
        return np.array([clf.rmse(df) for clf in self.classifiers])

    def rank(self, trainings, tests):
        """
        Rank the classifiers according to average
        root-mean-squared error across the training and test datasets.

        :param trainings: list of training datasets
        :param tests: list of test datasets
        """
        logger.info("Ranking submodels")
        assert len(trainings) == len(tests), "Must pass the same number of test and train datasets!"
        rmse = list()
        j = 1
        for train, test in zip(trainings, tests):
            logger.info("on training-test {}".format(j))
            self.fit(train)
            rmse.append(self.classifier_rmse(test))
            j+=1
        self.ranks = ss.rankdata(np.mean(np.vstack(rmse), axis=0))
        return self

    def combine(self, df, refit=True, n_draws=1000, psi=1.2):
        """
        Returns combined predictionswith confidence
        upper and lower bands based on the weighting
        parameter passed.

        We want to remove the classifiers that don't converge
        from the RMSE calculation, and it will return a None.
        Then the weights are re-scaled so account for a different
        number of classifiers in the denominator.

        Ideally in the future we take out classifiers that don't
        converge altogether.

        :param df: data frame to use for fitting,
                    and predicting.
        :param refit: (bool) whether or not to
                    re-fit each of the classifiers
        :param n_draws: (int) number of draws to create
        :param psi: (float) psi-value for the weighting scheme,
                    larger values of psi create more rapidly
                    declining weighing functions
        """
        logger.info("Combining classifiers to create draws with psi {}.".format(psi))
        self.draws = list()
        logger.info("moving on to refit block.")
        if refit:
            logger.info("Refit == True")
            for clf in self.classifiers:
                clf.fit(df)
        # Take out the classifiers that don't converge from the calculation of weights
        logger.info("next is valid classifiers.")
        self.valid_classifiers = [clf for clf in self.classifiers if clf.mdf is not None]
        logger.info("Next is weights....")
        self.weights = (psi**(len(self.valid_classifiers)-self.ranks)) / float(sum(psi**(len(self.valid_classifiers)-self.ranks)))
        logger.info("Next is new_weights...")
        new_weights = []
        for i in range(len(self.classifiers)):
            if self.classifiers[i].mdf is not None:
                logger.info("Appending weights")
                new_weights.append(self.weights[i])
            else:
                logger.info("Weights is None!!!")
                new_weights.append(0)
        logger.info("Self-weights is new_weights")
        self.weights = new_weights
        logger.info("Now looking at draws ints")
        for clf, weight in zip(self.classifiers, self.weights):
            if int(weight*n_draws) == 0:
                logger.info("Don't want to continue since no draws.")
                continue
            else:
                logger.info("Now we have draws!")
                self.draws.append(clf.generate_draws(df, n_draws=int(weight*n_draws), refit=refit))
        logger.info("Vstacking the draws.")
        self.draws = np.vstack(self.draws).T
        logger.info("Computing the estimate.")
        self.estimate = pd.DataFrame(self.draws).mean(axis=1).values
        logger.info("Computing the confidence interval.")
        self.intervals = pd.DataFrame(self.draws).quantile(q=[0.025, 0.975], axis=0)
        return self

    def get_rmse(self, dfs, refit=True, n_draws=1000, response='ln_rate', psi=1.2):
        """
        Calculates the RMSE of self.estimate compared to a data frame and a
        response.

        :param dfs: list of data frames on which to calculate RMSE
        :param refit: (bool) whether or not to re-fit the classifiers on the
                    dataset(s) being passed
        :param n_draws: (int) number of draws to produce
        :param response: (str) one of 'ln_rate' or 'lt_cf'
        :param psi: (float) parameter governing weighting scheme
        """
        rmse = []
        for df in dfs:
        	self.combine(df=df, refit=refit, n_draws=n_draws, psi=psi)
        	rmse.append(np.nanmean((self.estimate - df[response].values)**2)**0.5)
        self.rmse = np.nanmean(np.array(rmse))
        return self

    def plot_predictions(self, df, querystring, ages, response):
        """
        Plot time-series predictions and raw data for a given location
        and a list of ages given.

        :param df: data frame on which to make the predictions
        :param querystring: (str) query to run on pandas df to
                            subset the dataframe (ideally one location)
        :param ages: list of ages to include
        :param response: (str) one of 'ln_rate' or 'lt_cf'
        """
        df['predictions'] = self.estimate
        df_sub = df.query(querystring).sort_values('year')
        fig, axes = plt.subplots(1, len(ages), figsize=(20, 10), facecolor='white', sharey='row')
        i = 0
        for axis in axes:
            sub = df_sub.query('age == {}'.format(ages[i]))
            age_group_name = sub.age_group_name.unique()[0]
            location_name = sub.location_name.unique()[0]
            axis.plot(sub['year'], sub[response], 'ro')
            axis.plot(sub['year'], sub['predictions'])
            axis.set_title(age_group_name)
            if response == "ln_rate":
                ylabel = "Log Death Rate"
            else:
                ylabel = "Logit Cause Fraction"
            if i == 0:
                axis.set_ylabel(ylabel)
            i+=1
        plt.suptitle('Predictions for {}'.format(location_name))
        return fig
