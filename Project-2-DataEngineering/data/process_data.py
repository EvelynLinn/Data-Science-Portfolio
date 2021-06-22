import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Read the CSV file that contains message data
    messages = pd.read_csv(messages_filepath)
    # Read the CSV file that contains category data
    categories = pd.read_csv(categories_filepath)
    # Merge the messages and categories datasets using the common id
    df = pd.merge(messages, categories, on = 'id')
    return df

def clean_data(df):
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = []
    for i in range(0, len(row)):
        category_colnames.append(row[i][:-2])
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # Set each value to be the last character of the string
        for i in range(0, categories.shape[0]):
            categories[column][i] = categories[column][i].split('-')[-1]
            # Convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Replace categories column in df with new category columns
    df.drop(columns=['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    # Clean nonbinary data
    df = df[df.related != 2]
    # Remove duplicates
    df = df.drop_duplicates()
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('etl_pipeline', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
