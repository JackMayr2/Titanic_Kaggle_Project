# Kaggle Titanic Project

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# if stand-alone dataset
## api.dataset_download_files('kazanova/sentiment140',
##           file_name = 'training.1600000.processed.noemoticon.csv')
# example : https://www.kaggle.com/datasets/kazanova/sentiment140

# if competition dataset
# download to the project directory as a zip file
api.competition_download_files('titanic')
# create a folder in the project directory and download to it as a zip file
api.competition_download_files('titanic', 'data_titanic')
# example : https://www.kaggle.com/c/titanic

import zipfile

with zipfile.ZipFile('data_titanic/titanic.zip', 'r') as zipref:
    zipref.extractall('data_titanic/')
