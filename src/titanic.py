import numpy as np


def status_of_others_sharing_factor(shared_factor, df_train, df_test=None):
    """ Calculates if others sharing the same name/cabin/ticket survived

    Args:
        shared_factor (str): Factor shared with other passengers.
        df_train (pd.DataFrame): Dataframe with 'Survived' column to pull
            information from.
        df_test (pd.DataFrame, optional): Dataframe to test against the train
            dataframe. If `None`, the train dataframe is used.

    """

    if df_test is None:
        df_test = df_train

    # Cache the shared factor for all survivors.
    survived = df_train.loc[df_train['Survived'] == 1, shared_factor]
    died = df_train.loc[df_train['Survived'] == 0, shared_factor]

    # Determine how many passengers with the same factor survived, and how many
    # died. Do not include the passenger that is being compared.
    others_survived = df_test.apply(lambda r: ((survived.index != r.name) & (survived == r[shared_factor])).sum(), axis=1)
    others_died = df_test.apply(lambda r: ((died.index != r.name) & (died == r[shared_factor])).sum(), axis=1)

    # Determine if most people died, survived, or an equal amount of people died
    # and survived.
    status_of_others = np.sign(others_survived - others_died).replace(
        {-1: 'most_died', 0: 'equal_or_NaN', 1: 'most_survived'}
    )

    return status_of_others
