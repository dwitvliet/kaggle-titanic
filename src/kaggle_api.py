import os
import time
import logging

import kaggle


def get_recent_submission(competition):
    submissions = kaggle.api.competition_submissions(competition)
    return sorted(submissions, key=lambda s: s.ref)[-1]


def get_recent_submission_score(competition):
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
