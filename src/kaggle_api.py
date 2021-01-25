import os
import time
import logging

import kaggle


def get_recent_submission(competition):
    """ Get the most recent submission.

    Args:
        competition (str): Kaggle competition to get the submission for.

    Returns:
        Submission object with attributes `status` and `publicScore`.

    """
    submissions = kaggle.api.competition_submissions(competition)
    return sorted(submissions, key=lambda s: s.ref)[-1]


def get_recent_submission_score(competition):
    """ Get the score of the most recent submission.

    Wait until the submission status is no longer pending (for a maximum of 60
    seconds), then return the submission score.

    Args:
        competition (str): Kaggle competition to get the submission for.

    Returns:
        float

    """
    max_wait = 60

    while max_wait > 0:
        submission = get_recent_submission(competition)
        if submission.status != 'pending':
            break
        time.sleep(1)
        max_wait -= 1

    if submission.status == 'error':
        logging.error(f'Submission error:\n{submission.errorDescription}')
        return 0

    return submission.publicScore


def submit(competition, series, description):
    """

    Args:
        competition (str): Kaggle competition to get the submission for.
        series (pd.Series): Series containing all predictions as values and all
            test ids as index. The name of the index and series becomes the
            header for the submission file.
        description (str): Description of submission.

    Returns:
        float: Submission score.

    """

    # Create directory to save submission if it does not exist.
    if not os.path.exists('submissions'):
        os.mkdir('submissions')
    dir_path = f'submissions/{competition}'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # Count previous submissions to determine index.
    index = len(kaggle.api.competition_submissions(competition))

    # Save submission to file.
    file_path = os.path.join(dir_path, f'{index}.csv')
    series.to_csv(file_path)

    # Submit.
    kaggle.api.competition_submit(file_path, description, competition, quiet=True)
    return get_recent_submission_score(competition)
