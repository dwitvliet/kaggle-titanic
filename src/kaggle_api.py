import os

import kaggle


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
    kaggle.api.competition_submit(file_path, description, competition)
