import numpy as np


def status_of_others_sharing_factor(shared_factor, df_ref, df_test=None):


    if df_test is None:
        df_test = df_ref

    # Cache the shared factor for all survivors.
    survived = df_ref.loc[df_ref['Survived'] == 1, shared_factor]
    died = df_ref.loc[df_ref['Survived'] == 0, shared_factor]

    others_survived = df_test.apply(lambda r: ((survived.index != r.name) & (survived == r[shared_factor])).sum(), axis=1)
    others_died = df_test.apply(lambda r: ((died.index != r.name) & (died == r[shared_factor])).sum(), axis=1)

    status_of_others = np.sign(others_survived - others_died).replace(
        {-1: 'most_died', 0: 'equal_died_and_survived', 1: 'most_survived'}
    )

    return status_of_others
