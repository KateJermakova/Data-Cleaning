import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import datetime

df = pd.read_csv('C:/Users/kater/Downloads/ODI-2023.csv', sep=';')

#In the columns with numeric values check if there are non-numeric ones and if replace them with NaN
df['Have you taken a course on information retrieval?'] = pd.to_numeric(df['Have you taken a course on information retrieval?'], errors='coerce')
df['How many students do you estimate there are in the room?'] = pd.to_numeric(df['How many students do you estimate there are in the room?'], errors='coerce')
df['What is your stress level (0-100)?'] = pd.to_numeric(df['What is your stress level (0-100)?'], errors='coerce')
df['Give a random number'] = pd.to_numeric(df['Give a random number'], errors='coerce')

missing_cols = ['Have you taken a course on information retrieval?', 'How many students do you estimate there are in the room?',
                'What is your stress level (0-100)?', 'Give a random number']
# Create a copy of the DataFrame with the missing values replaced by NaN
df_missing = df.copy()
df_missing[missing_cols] = df_missing[missing_cols].replace('?', pd.np.nan)

# Create the imputer object
imputer = SimpleImputer(strategy='median')
# Perform the regression imputation
df_imputed = pd.DataFrame(imputer.fit_transform(df_missing[missing_cols]), columns=missing_cols)
df[missing_cols] = df_imputed[missing_cols]


#Make birth dates in one format
from dateutil import parser
formatted_dates = []
for date in df['When is your birthday (date)?']:
    try:
        dt = parser.parse(date)
        formatted_dates.append(dt.strftime('%Y-%m-%d'))
    except:
        formatted_dates.append(None)
        
# replace missing values with default date
default_date = '1998-01-01'
formatted_dates = [date if date is not None else default_date for date in formatted_dates]
df['When is your birthday (date)?'] = formatted_dates

# removing the years that has extreme values
df['When is your birthday (date)?'] = pd.to_datetime(df['When is your birthday (date)?'], errors='coerce')
year = 2007
df = df[df['When is your birthday (date)?'].dt.year <= year]


#making time to one format
formatted_times=[]
#define the different time formats in the dataset
formats = ["%H:%M", "%H.%M", "%H", "%I:%M%p", "%I%p", "%I.%M%p", "%I.%M %p", "%I %p"]

# function to convert a str to a datetime object, or return Nan
def parse_time(time_string):
    # Check for common errors in the time string
    if time_string in ['12:00', '12.00', '12']:
        return '00:00'
    if time_string in ['12:30', '12.30']:
        return '00:30'
    if time_string in ['11:00', '11.00', '11']:
        return '23:00'
    if time_string in ['11:30', '11.30']:
        return '23:30'
    if time_string in ['10:00', '10.00', '10']:
        return '22:00'
    if time_string in ['10:30', '10.30']:
        return '22:30'
    if time_string in ['9:00', '9.00', '9']:
        return '21:00'
    if time_string in ['9:30', '9.30']:
        return '21:30'
    if "." in time_string:
        time_string = time_string.replace(".", ":")
    if "pm" in time_string.lower() and ":" not in time_string:
        time_string = time_string.lower().replace("pm", ":00pm")
    if "am" in time_string.lower() and ":" not in time_string:
        time_string = time_string.lower().replace("am", ":00am")

    for format in formats:
        try:
            time_obj = datetime.datetime.strptime(time_string, format)
            # Convert to a 24-hour format
            return time_obj.strftime("%H:%M")
        except ValueError:
            pass
    # If none of the formats work, return NaN
    return float("NaN")

# apply the parse_time function 
df['Time you went to bed Yesterday'] = df['Time you went to bed Yesterday'].apply(parse_time)
# imputation of missing values of time using mode 
imputer = SimpleImputer(strategy='most_frequent')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


#upper bond for stress level is given in the question (=100)
df['What is your stress level (0-100)?'] = df['What is your stress level (0-100)?'].clip(upper=100.0)
#lower bond for stress level is given in the question (=0)
df['What is your stress level (0-100)?'] = df['What is your stress level (0-100)?'].clip(lower=0)
#lower bond for the number of students in the room, it can't be negative
df['How many students do you estimate there are in the room?'] = df['How many students do you estimate there are in the room?'].clip(lower=0)

#removing all non numeric values from sport hours column and replacing with the most frequesnt value
if not df.iloc[:, 12].apply(lambda x: str(x).isnumeric()).all():
    # replace non numeric values with NaN
    df.iloc[:, 12] = pd.to_numeric(df.iloc[:, 12], errors='coerce')
    # replace NaN values with mode
    df.iloc[:, 12] = imputer.fit_transform(df.iloc[:, 12].values.reshape(-1, 1))
    # convert back to int
    df.iloc[:, 12] = df.iloc[:, 12].astype(int)

#Editing column of programme
df['What programme are you in?'] = df['What programme are you in?'].str.lower()
df['What programme are you in?'] = df['What programme are you in?'].str.replace('artificial intelligence', 'ai')

df['Have you taken a course on machine learning?'].replace('unknown', np.nan, inplace=True)
df['Have you taken a course on statistics?'].replace('unknown', np.nan, inplace=True)
df['Have you taken a course on databases?'].replace('unknown', np.nan, inplace=True)
df.iloc[:, 10].replace('unknown', np.nan, inplace=True)
# fit the imputer on the column and transform the column in-place
df['Have you taken a course on statistics?'] = imputer.fit_transform(df[['Have you taken a course on statistics?']])
df['Have you taken a course on machine learning?'] = imputer.fit_transform(df[['Have you taken a course on machine learning?']])
df['Have you taken a course on databases?'] = imputer.fit_transform(df[['Have you taken a course on databases?']])
df.iloc[:, 10] = imputer.fit_transform(df.iloc[:, 10].values.reshape(-1, 1))
df.iloc[:, 10] = df.iloc[:, 10].astype(str)

# Text preprocessing using NLP and categorize by topics columns

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    # remove punctuation and convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    # remove words with fewer than 3 characters
    tokens = [token for token in tokens if len(token) > 2]
    # remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # lemmatize 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, pos='v') for token in tokens]
    # stem tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    # join tokens back into text and return it
    text = ' '.join(tokens)
    return text

df['What makes a good day for you (1)?'] = df['What makes a good day for you (1)?'].apply(preprocess_text)
df['What makes a good day for you (2)?'] = df['What makes a good day for you (2)?'].apply(preprocess_text)

#edit columns 'What makes a good day for you'
# create a dictionary to map the labels to the words
label_map = {('weather', 'sun','sunni', 'morn', 'sunshin'): 'weather', 'family': 'family', ('friends', 'game','people','fun', 'company'): 'friends', 
             ('gym', 'sports', 'sport'): 'sport', ('music', 'concert', 'piano'): 'music', ('good food', 'meal', 'coff', 'beer'): 'food', 
             ('relax', 'sleep', 'rest','sex'): 'sleep', ('work', 'class', 'lectur', 'product', 'code','time', 'free', 'stress'): 'no obligations'}


def get_label(row):
    programme = row['What makes a good day for you (1)?'].lower()
    for key in label_map:
        if isinstance(key, tuple):
            for keyword in key:
                if keyword in programme:
                    return label_map[key]
        else:
            if key in programme:
                return label_map[key]
    return 'unknown'

df['label'] = df.apply(get_label, axis=1)
# print out the labels
#print(df['label'])
#unknown values count
count_programme = df['label'].value_counts()['unknown']
#print(count_programme)



#Edit the study programme column
# create a dictionary to map the labels to the words
label_map_programme = {'ai': 'ai', ('computer science', 'cs'): 'computer science', 'bioinformatics': 'bioinformatics', 
             'business analytics': 'business analytics', 'econometrics': 'econometrics', 'finance': 'finance and technology', 
             'data': 'data science', 'quantitative risk management': 'quantitative risk management',
              'human language technology': 'human language technology', 'phd': 'phd', 'exchange': 'exchange'}

def get_label_programme(row):
    programme = row['What programme are you in?'].lower()
    for key in label_map_programme:
        if isinstance(key, tuple):
            for keyword in key:
                if keyword in programme:
                    return label_map_programme[key]
        else:
            if key in programme:
                return label_map_programme[key]
    return 'other'

df['label_programme'] = df.apply(get_label_programme, axis=1)
#print(df['label_programme'])

#count ''other'' values in label column
count_programme = df['label_programme'].value_counts()['other']
#print(count_programme)
