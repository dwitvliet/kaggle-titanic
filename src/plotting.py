import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection


def learning_curve(estimator, X, y, ax=None, cv=None, n_jobs=None,
                   train_sizes=np.linspace(.1, 1.0, 5), legend=True):
    """ Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    if cv is None:
        cv = model_selection.ShuffleSplit(n_splits=100, test_size=0.2)

    ax.set_xlabel('Training examples')
    ax.set_ylabel('Score')

    train_sizes, train_scores, test_scores, fit_times, _ = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    ax.grid()
    ax.fill_between(
        train_sizes, train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, alpha=0.1,
        color='r'
    )
    ax.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, alpha=0.1,
        color='g'
    )
    ax.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
    ax.plot(train_sizes, test_scores_mean, 'o-', color='g', label='CV score')
    if legend:
        ax.legend(loc='best')

    return ax
