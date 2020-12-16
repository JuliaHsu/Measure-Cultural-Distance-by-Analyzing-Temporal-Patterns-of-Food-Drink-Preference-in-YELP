import pandas as pd
import numpy as np
import spacy
from nltk.tokenize import RegexpTokenizer
import gensim
from nltk.stem import PorterStemmer
import gensim.models.keyedvectors as word2vec
DATA = '../yelp_dataset/'
FEATURES_DATA = '../yelp_dataset/features/'
JOIN_TABLE = '../yelp_dataset/join_table/'
RESTAURANT_DATA = '../yelp_dataset/restaurant_data/'
RESULT = '../result/'
def read_rest_review():
    rest_review = pd.read_csv(JOIN_TABLE + 'restaurant_review.csv', dtype = object)
    # print(rest_review)
    rest_review = rest_review[['business_id','review_id','name','categories','text','date']]
    # print(rest_review)
    rest_root_cat_df = get_rest_root_cat()
    rest_review = pd.merge(rest_review,rest_root_cat_df,on='business_id') 
    processed_review = text_preprocess(rest_review)
    return processed_review

def get_rest_root_cat():
    rest_root_cat_df = pd.read_csv(RESULT + 'root/business_category/kmeans/restaurant_root_cat_kmean_filtered_15.csv')
    rest_root_cat_df = rest_root_cat_df[['business_id','category']]
    return rest_root_cat_df
def text_preprocess(rest_review):
    rest_review = remove_non_alphbet(rest_review)
    rest_review = get_word_token(rest_review)
    rest_review = remove_stop_words(rest_review)
    category_ls = list(set(rest_review['category'].tolist()))
    category_ls.sort()
    processed_rest_review = pd.DataFrame()
    for cat in category_ls:
        rest_cat_review = rest_review.loc[rest_review['category'] == cat]
        rest_cat_review = get_bigrams(rest_cat_review)
        processed_rest_review = processed_rest_review.append(rest_cat_review)
    print(processed_rest_review)
    processed_rest_review = lemmatize_word(processed_rest_review)
    
    # rest_review = stemming_word(rest_review)

    # rest_review['processed_text'] = rest_review.apply(lambda x: x['categories'].split(', ') + x['text'], axis = 1)
    # rest_review['processed_text'] = rest_review.apply(lambda x: x['categories'] + x['text'], axis = 1)
    processed_rest_review.to_csv(RESTAURANT_DATA+'processed_review.csv')
    print(processed_rest_review)
    return processed_rest_review
    
def remove_non_alphbet(rest_review):
    # Apply a function along an axis (default: column) of the DataFrame.
    # lower case
    rest_review[['text']] = rest_review[['text']].apply(lambda col: col.str.lower())
    # remove non alphabets (digits, special characters...)
    rest_review[['text']] = rest_review[['text']].apply(lambda col: col.str.replace(r'[^a-z\_\- ]', ''))
    # remove multiple whitespaces
    rest_review[['text']] = rest_review[['text']].apply(lambda col: col.str.replace('\s+',' '))
    # remove leading and trailing whitespaces including tabs
    rest_review[['text']] =  rest_review[['text']].apply(lambda col: col.str.strip())
   
    rest_review.loc[rest_review['text'].isnull(), 'text'] = ""
    # print(rest_review)
    return rest_review

def get_word_token(rest_review):
    print("get_word_token\n")
    tokenizer = RegexpTokenizer(r'\w+')
    # rest_review['categories'] = rest_review.apply(lambda x: tokenizer.tokenize(x['categories']), axis = 1)
    rest_review['text'] = rest_review.apply(lambda x: tokenizer.tokenize(x['text']), axis = 1)
    return rest_review

def remove_stop_words(rest_review):
    print("remove_stop_words\n")
    sp = spacy.load('en')
    all_stopwords = sp.Defaults.stop_words
    # join the list of words to lemmatize the words
    # rest_review['categories'] = rest_review.apply(lambda x: " ".join( [word for word in x['categories'] ] ), axis = 1)

    # join the list of words to lemmatize the words
    # rest_review['text'] = rest_review.apply(lambda x: " ".join( [word for word in x['text'] if not word in all_stopwords ] ), axis = 1)
    # tokenize for bigram
    rest_review['text'] = rest_review.apply(lambda x: [word for word in x['text'] if not word in all_stopwords ] , axis = 1)
    return rest_review

def get_bigrams(rest_review):
    print("bigrams\n")
    data_words =rest_review['text']
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    row =0
    # print([bigram_mod[content] for content in data_words])
    for content in data_words:
        rest_review['text'].iloc[row] = bigram_mod[content]
        row+=1
    # join the list of words to lemmatize the words
    rest_review['text'] = rest_review.apply(lambda x: " ".join(x['text']), axis = 1)
    return rest_review

def lemmatize_word(rest_review):
    print("lemmatize_word\n")
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # rest_review['categories'] = rest_review.apply(lambda x: [ token.lemma_ for token in nlp(x['categories']) ], axis = 1)
    rest_review['text'] = rest_review.apply(lambda x: [ token.lemma_ for token in nlp(x['text']) ], axis = 1)
    # remove words that have less than 2 characters 
    rest_review['text'] = rest_review.apply(lambda x: [ word for word in x['text'] if len(word) >2], axis = 1 )
    return rest_review

def stemming_word(rest_review):
    rest_review['categories'] = rest_review.apply(lambda x: x['categories'].split(', '),axis =1)
    ps = PorterStemmer() 
    row = 0
    for cat in rest_review['categories']:
        stemmed_ls = []
        for word in cat:
            stemmed_word = ps.stem(word) 
            stemmed_ls.append(stemmed_word)
        rest_review['categories'].iloc[row] = stemmed_ls
        row+=1
   
    row = 0
    for review in rest_review['text']:
        stemmed_ls = []
        for word in review:
            stemmed_word = ps.stem(word) 
            stemmed_ls.append(stemmed_word)
        rest_review['text'].iloc[row] = stemmed_ls
        row+=1
    return rest_review

def get_word_freq(processed_rest_review):
    print("get_word_freq\n")
    text_ls = processed_rest_review['text'].to_list()
    word_ls = []
    for ls in text_ls:
        for w in ls:
            word_ls.append(w)
    word_df = pd.DataFrame (word_ls,columns=['word'])
    word_freq = word_df['word'].value_counts()
    word_freq.sort_values(ascending = False, inplace = True)
    word_freq.to_csv(FEATURES_DATA + "word_freq_review.csv")
    most_freq = word_freq.head(1000)
    most_freq.to_csv(FEATURES_DATA + "1000_freq_word_review.csv")
    print(word_freq)
    print(most_freq)
    
    

processed_rest_review = read_rest_review()
get_word_freq(processed_rest_review[['text']])
# processed_rest_review = pd.read_csv(RESTAURANT_DATA + 'processed_review.csv', index_col = 0, converters={'text': lambda n: n[1:-1].replace("'","").split(', ')}) 
# get_word_freq(processed_rest_review[['text']])


