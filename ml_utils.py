#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Utility module to facilitate searching through hyperparameter 
space without having to write a lot of boiler plate code.

Example:
search = random_search(X, y, estimator, params, n_iter, scoring, cv, n_jobs=1, **kwargs)
or 
search = grid_search(X, y, estimator, params, scoring, cv, n_jobs=1, **kwargs)
Then, check results:
display_summary(search, show_report=True, show_graphics=True, n_top=5)   
"""

import os
import sys
import time
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skopt import gp_minimize
from sklearn.model_selection import (GridSearchCV, 
                                     RandomizedSearchCV, 
                                     learning_curve, 
                                     validation_curve, 
                                     cross_val_score)
from sklearn.exceptions import NotFittedError
from operator import itemgetter


def bayes_search(X, y, estimator, params, scoring, cv, n_calls, greater_is_better, n_jobs=1, fit_estimator=True, **kwargs):
    """
    Leverage package `scikit-optimize` (skopt) in order to do 
    hyperparameter tuning in a Bayesian optimalization framework
    
    Parameters:
    -------------------
     X : training data (n_samples, n_features)
     y : training labels (n_samples,)
     estimator : estimator implementing `fit` and `predict` methods 
     params : parameter dictionary  
     scoring : scoring (string or callable)
     cv : cross-validation generator (see docstring of e.g. RandomizedSearchCV)
     n_calls : number of iterations to run Bayesian optimization
     greater_is_better : set to True if optimizing a metric that should 
         be increased for better model performance (e.g. accuracy). Conversely,
         set to False if you are working with a metric that should be minimized.
     n_jobs : number of jobs to run in parallel (default: 1)
     fit_estimator : If True, refit the best estimator on all the data (default: True)
     **kwargs : other parameters you wish to send to `gp_minimize`
    
    Returns:
    -------------------
        Tuple consisting of (estimator, best_params, best_score) where `estimator`
        by default has been refitted (see `fit_estimator`) to all the input data.
        `best_params` is a dictionary with optimized parameters
        `best_score` is the best score obtained on the cross-validated data
    """
    if len(y.shape)>1: y = y.ravel()
    param_names = list(name for name,_ in params.items())
    dim = list(dm for _,dm in params.items())

    def evalfunc(parameters):
        try_params = {k:v for k, v in zip(param_names, parameters)}
        estimator.set_params(**try_params)        
        score = np.mean(np.array(cross_val_score(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)))
        return -score if greater_is_better else score

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            res_gp = gp_minimize(evalfunc, dim, n_calls=n_calls, **kwargs)  
        except:
            warnings.warn("Optimization failed: %s" % sys.exc_info()[1], NotFittedError)
            return None
        
    best_score = -res_gp.fun if greater_is_better else res_gp.fun
    best_params = {k:v for k,v in zip(param_names, res_gp.x)}      
    if fit_estimator:
        estimator.set_params(**best_params)
        estimator.fit(X, y)    
    return (estimator, best_params, best_score)

def grid_search(X, y, estimator, params, scoring, cv, n_jobs=1, **kwargs):
    """ Convenience method for grid search through hyperparameter space of any estimator
    
    Input:
    --------------
        X: training data (n_samples, n_features)
        y: training labels (n_samples,)
        estimator: estimator implementing `fit` and `predict` methods 
        params: parameter dictionary  
        scoring: scoring (string or callable)
        cv: cross-validation generator (see docstring of RandomizedSearchCV)
        n_jobs: number of jobs to run in parallel (default: 1)
        **kwargs: other parameters you wish to send to RandomizedSearchCV
    
    Output:
    --------------
        Fitted instance of GridSearchCV or `None` if we failed to fit.
    """
    if len(y.shape)>1: y = y.ravel()
        
    search = GridSearchCV(estimator, param_grid=params, 
                          scoring=scoring, cv=cv, n_jobs=n_jobs, **kwargs)    
    start_time = time.time()
    try:
        search.fit(X, y)
    except:
        warnings.warn("Failed to fit.", NotFittedError)
        return None
    print("Total time spent searching: {0:.1f} minutes."\
          .format((time.time()-start_time)/60.), end='\n')
    return search 

def random_search(X, y, estimator, params, n_iter, scoring, cv, n_jobs=1, **kwargs):
    """ Convenience method for random search through hyperparameter space of any estimator
    
    Input:
    --------------
        X: training data (n_samples, n_features)
        y: training labels (n_samples,)
        estimator: estimator implementing `fit` and `predict` methods 
        params: parameter distribution dictionary 
        n_iter: number of random draws from parameter dictionary to test
        scoring: scoring (string or callable)
        cv: cross-validation generator (see docstring of RandomizedSearchCV)
        n_jobs: number of jobs to run in parallel (default: 1)
        **kwargs: other parameters you wish to send to RandomizedSearchCV
    
    Output:
    --------------
        Fitted instance of RandomizedSearchCV or `None` if we failed to fit.
    """
    if len(y.shape)>1: y = y.ravel()
        
    search = RandomizedSearchCV(estimator, param_distributions=params, n_iter=n_iter, 
                                scoring=scoring, cv=cv, n_jobs=n_jobs, **kwargs)    
    start_time = time.time()
    try:
        search.fit(X, y)
    except:
        warnings.warn("Failed to fit.", NotFittedError)
        return None
    print("Total time spent searching: {0:.1f} minutes."\
          .format((time.time()-start_time)/60.), end='\n')
    return search 

def display_summary(search, show_report=True, show_graphics=True, n_top=5):
    """ 
    Display summary of fitting procedure   
    Input:
    --------------
        search: fitted instance of RandomizedSearchCV (output from `random_search`)
        show_report: display a text report (for `n_top` models)
        show_graphics: display a graphical summary of fitted models (all)
        n_top: show model summary for `n_top` models (default: 5)
        
    Ouput:
    --------------
        Nothing to output
    """
    assert hasattr(search, 'cv_results_')
    df = pd.DataFrame(search.cv_results_)
    # Print a report
    if show_report: 
        _display_report(df, n_top)        
    # Visualize graphically
    if show_graphics:
        _display_graphics(df)        
    return
    
def _display_report(df, n_top):
    top_scores = df.sort_values('rank_test_score').head(n_top)
    for i, (_, score) in enumerate(top_scores.iterrows()):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_test_score, score.std_test_score))
        print("Parameters: {0}".format(score.params), end='\n\n')
    return

def _display_graphics(df, **kwargs):
    plt.figure(figsize=(12,4))
    plt.errorbar(x=1+df.index, y=df.mean_test_score, yerr=df.std_test_score, fmt='--o', ecolor='k', **kwargs)
    plt.xticks(1+df.index)
    plt.xlabel("Model index"); plt.ylabel("CV score"); plt.title("Results from cross-validation")
    plt.tight_layout()
    return plt
              
def plot_learning_curve(title, X, y, estimator, scoring, cv, n_jobs=1, ylim=None, train_sizes=np.linspace(.1, 1.0, 5), **kwargs):
    """
    Visualize learning curve. Helps to evaluate under-/overfitting. 
    
    Parameters:
    --------------
        title : Title for the chart
        X: training data (n_samples, n_features)
        y: training labels (n_samples,)
        estimator: estimator to evaluate behavior for. Implements `fit` and `predict` methods
        scoring: type of metric to evaluate
        cv: cross-validation generator
        n_jobs: number of jobs to run in parallel
        ylim: tuple, shape (ymin, ymax), optional. Defines minimum and maximum yvalues plotted
        train_sizes: proportions of training data to inspect (default: np.linspace(0.1, 1.0, 5))
        **kwargs: other arguments to send to `learning_curve` method
    """
    plt.figure(figsize=(12,4))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, scoring=scoring, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, **kwargs)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std, alpha=0.1, color="g")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

def explore_param(title, X, y, estimator, scoring, cv, param_name, param_range, n_jobs=1, ylim=None, logparam=False, **kwargs):
    """
    Visualize a 1-dimensional validation curve for a given parameter. Useful when checking
    sensitivity to a given parameter (evaluate conditional distribution).
    
    Parameters:
    --------------
        title : title for the chart
        X: training data (n_samples, n_features)
        y: training labels (n_samples,)
        estimator: estimator to evaluate behavior for. Implements `fit` and `predict` methods
        scoring: type of metric to evaluate
        cv: cross-validation generator
        param_name: name of parameter to vary
        param_range: parameter range
        n_jobs: number of jobs to run in parallel
        ylim: tuple, shape (ymin, ymax), optional. Defines minimum and maximum yvalues plotted
        logparam: If True, then plot logarithmic x-axis (good for parameters with loguniform priors)
        **kwargs: other arguments to send to `validation_curve` method
    """    
    train_scores, test_scores = validation_curve(estimator, X, y, 
                                                 param_name=param_name, param_range=param_range,
                                                 cv=cv, scoring=scoring, n_jobs=n_jobs, **kwargs)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(12,4))
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Score")
    if ylim is not None:
        plt.ylim(*ylim)
    
    if logparam:
        plt.semilogx(param_range, train_scores_mean, label="Training score", color="darkorange", lw=2)
    else:
        plt.plot(param_range, train_scores_mean, label="Training score", color="darkorange", lw=2)
    plt.fill_between(param_range, train_scores_mean-train_scores_std,
                     train_scores_mean+train_scores_std, alpha=0.2, color="darkorange", lw=2)
    if logparam:
        plt.semilogx(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=2)
    else:
        plt.plot(param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=2)
    plt.fill_between(param_range, test_scores_mean-test_scores_std,
                     test_scores_mean+test_scores_std, alpha=0.2, color="navy", lw=2)
    plt.legend(loc="best")
    return plt