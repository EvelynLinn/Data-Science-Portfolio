# Disaster Response Pipeline Project
### Installations
#### For Data Processing
import pandas as pd
from sqlalchemy import create_engine
#### For Machine Learning pipeline# General libraries
```
import re
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import  create_engine

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import pickle
```
### Files Explained
The files for this project are structured as follows within the folder Disaster-pipelines:
In the data processing step:
- ETL Pipeline Preparation.ipynb
- process_data.py
- messages.csv
- categories.csv
- etl_pipeline.db
In the machine learning step:
- ML Pipeline Preparation.ipynb
- train_classifier.py
- classifier.pkl
In the web app step:
- run.py
- master.html
- go.html

### Instructions for Running
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Web App Interface
