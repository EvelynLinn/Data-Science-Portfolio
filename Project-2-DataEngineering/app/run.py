import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
#from train_classifier import DenseTransformer

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()
        
app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../etl_pipeline.db')
#engine = create_engine('sqlite:///../data/etl_pipeline.db')
df = pd.read_sql_table('etl_pipeline', engine)

# load model
model = joblib.load("../classifier.pkl")
#model = joblib.load("../models/classifier.pkl") #original


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    # for second visualization
    df_for_graph = df[df.columns[4:]]
    category_counts=df_for_graph.sum()
    category_names = df_for_graph.columns.values #list(category_counts.index)
    import plotly.graph_objects as go
    genres = ['Direct', 'News', 'Social']
    # for third visualization
    related_counts = df[df['related']==1].groupby('genre').count()['related']
    aid_counts = df[df['aid_related']==1].groupby('genre').count()['aid_related']
    direct_counts = df[df['direct_report']==1].groupby('genre').count()['direct_report']
    #horizontal bar graph of category distribution
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
    # second graph
            {'data': [
                Bar(
                    y=category_names,
                    x=category_counts,
                    orientation ='h',
                    marker=dict(color='LightSkyBlue',
                    line=dict(color='MediumPurple')
                )
                )
            ],
            'layout': {
                'title': 'Distribution of categories',
                'yaxis': {
                'title': "Categories" #, 'tickangle': 30, 'dtick':1
            },
                'xaxis': {
                    'title': "Count"
            }
        }
    },
    # third graph
            go.Figure(data=[
            go.Bar(name='Related', x=genres, y=related_counts),
            go.Bar(name='Aid Related', x=genres, y=aid_counts),
            go.Bar(name='Direct Report', x=genres, y=direct_counts)
        ])

    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
