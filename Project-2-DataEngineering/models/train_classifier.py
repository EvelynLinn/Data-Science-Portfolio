import sys
import re
import numpy as np
import pandas as pd
import sqlalchemy
from sqlalchemy import  create_engine
import pickle
# NLTK libraries
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# Scikit-learn libraries
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
from sklearn.naive_bayes import GaussianNB


def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('etl_pipeline', con=engine)
    # Cleaning data
    df = df[df.related != 2]
    X = df.message.values
    Y = df[df.columns[4:]]
    category_names = Y.columns.values
    return X, Y, category_names



def tokenize(text):
    '''
    This function first normalizes the plain texts, detects URLs in the messages, replace those URLs with a place holder;
    It further tokenizes words and removes stop words;
    After maps and lemmatizes different versions of same words to their root forms,
    the function finally returns the clean tokens.
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    text = text.lower()
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    word_tokens = word_tokenize(text)

    words = [w for w in word_tokens if w not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


def build_model():
    model = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer(norm='l1')),
    ('to_dense', DenseTransformer()),
    ('clf', MultiOutputClassifier(GaussianNB()))
    ])
    return model

def evaluate_model(model, X_test, Y_test, categories):
    #model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    for i, c in enumerate(categories):
        print(c)
        print(classification_report(Y_test.iloc[:,i] , Y_pred[:,i]))


def save_model(model, model_filepath):
    final_model = 'finalized_model.sav'
    pickle.dump(model, open(final_model, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
