from turtle import pd
from deep_translator import GoogleTranslator
import en_core_web_sm
import spacy
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy import displacy
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from autocorrect import Speller
import string
import re
import datefinder
import nltk
from nltk.corpus import wordnet
from jellyfish import soundex, metaphone, nysiis, match_rating_codex,\
    levenshtein_distance,\
    jaro_similarity
from itertools import groupby
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from tkinter import *

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def stem_words(txt):
    stems = [stemmer.stem(word) for word in txt]
    return stems


def lemmatize_word(txt):
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in txt]
    return lemmas


def tf_idf(search_keys, Docs):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_weights_matrix = tfidf_vectorizer.fit_transform(Docs)
    search_query_weights = tfidf_vectorizer.transform([search_keys])
    return search_query_weights, tfidf_weights_matrix


def cos_similarity(search_query_weights, tfidf_weights_matrix):
    cosine_distance = cosine_similarity(search_query_weights, tfidf_weights_matrix)
    similarity_list = cosine_distance[0]
    return similarity_list


def most_similar(similarity_list, min_Doc=1):
    most_similar = []
    while min_Doc > 0:
        tmp_index = np.argmax(similarity_list)
        most_similar.append(tmp_index)
        similarity_list[tmp_index] = 0
        min_Doc -= 1
    return most_similar

def K_Means(tfidf_weights_matrix,num=5):
    kmeans = KMeans(n_clusters= num, init='k-means++', max_iter = 500, n_init = 1)
    kmeans.fit(tfidf_weights_matrix)
    print(kmeans.cluster_centers_)
    return kmeans

def calculate_precision(res, gold_standard):
    true_pos = 0
    for item, x in res.items():
        for i, j in mappings.items():
            if i == '01':
                for k in j:

                    if k == str(item):
                        print(k)
                        true_pos += 1
    print(true_pos)
    return float(true_pos) / float(len(res.items()))


def delete():
    entry.delete(0,END) 

def backspace():
    entry.delete(len(entry.get())-1,END) 

def submit(datasetTarget): 
    print("datasetTarget: ", datasetTarget)
    # query
    file = open("common_words", "r")
    fileData = file.read()
    file.close()
    stopwords = re.findall("\S+", fileData)
    query = entry.get() 
    translated = GoogleTranslator(source='ar', target='en').translate(query)  
    print("translated: ", translated)
    # auto correct
    spell = Speller(lang='en')
    Query = spell(translated)
    print("spell Query: ", Query)
    # split into words
    tokens = word_tokenize(Query)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # filter out stop words
    stop_words = set(stopwords)
    words = [w for w in stripped if not w in stop_words]
    # stemming
    stem_word = stem_words(words)
    proccesedQuery = " ".join(stem_word)
    print(proccesedQuery)
    # text preprocessing start
    file = open("common_words", "r")
    fileData = file.read()
    file.close()
    stopwords = re.findall("\S+", fileData)
    if(datasetTarget == "first"):
        for x in range(1, 403666):
            # read from files
            print("*******************")
            print("Doc :", x)
            print("corpus1/{}.text".format(x))
            f = open("corpus1/{}.text".format(x), "r")
            text = f.read()
            f.close()
            ## SENT TOKENIZE and split up 
            sent_tokenize(text)
            ## split into words
            print("Orginal text :",text)
            tokens = word_tokenize(text)
            # print("Doc :",x)
            print("TOKENIZE:",tokens)
            ## convert to lower case then remove all punctuation except words and space
            tokens = [w.lower() for w in tokens]
            print("lower:",tokens)
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            print("remove punctuation:",stripped)
            ## remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]
            print("not alphabetic:",words)
            ## filter out stop words
            stop_words = set(stopwords)
            words = [w for w in words if not w in stop_words]
            print("filter stop words:",words)
            ## stemming
            stem_word = stem_words(words)
            print("stemming:",stem_word)
            ## lemmetization with pos
            def pos_tagger(nltk_tag):
                if nltk_tag.startswith('J'):
                    return wordnet.ADJ
                elif nltk_tag.startswith('V'):
                    return wordnet.VERB
                elif nltk_tag.startswith('N'):
                    return wordnet.NOUN
                elif nltk_tag.startswith('R'):
                    return wordnet.ADV
                else:
                    return None
            listToStr = ' '.join([str(elem) for elem in words])
            pos_tagged = nltk.pos_tag(nltk.word_tokenize(listToStr))
            print("pos_tagged :",pos_tagged)
            wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
            print("wordnet_tagged :",wordnet_tagged)
            lemmatized_sentence = []
            for word, tag in wordnet_tagged:
                if tag is None:
                    lemmatized_sentence.append(word)
                else:
                    lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            lemmatized_sentence = " ".join(lemmatized_sentence)
            print("lemmatized_sentence :",lemmatized_sentence)
            ### indexing, ranking, matching 
            text2 = proccesedQuery
            res,res2 = tf_idf(text2, words)
            print("search_query_weights ",res)
            print("tfidf_weights_matrix ", res2)
            res3 = cos_similarity(res,res2)
            print("similarity_list ", res3)
            res4 = most_similar(res3)
            print("most_similar ", res4)
            ### matches dates...etc 
            matches = datefinder.find_dates(text)
            # for match in matches:
                # print("match", match)
            spacy_model = spacy.load('en_core_web_sm')
            listToStr2 = ' '.join([str(elem) for elem in words])
            entity_doc = spacy_model(listToStr2)
            print([(entity.text, entity .label_) for entity in entity_doc.ents])
            ### soundex
            sounds_encoding_methods = [soundex, metaphone, nysiis, match_rating_codex]
            report = pd.DataFrame([words]).T
            report.columns = ['word']
            for i in sounds_encoding_methods:
                report[i.__name__] = report['word'].apply(lambda x: i(x))
            # print(report)
        
    else:
        for x in range(1, 5233329):
            # read from files
            print("*******************")
            print("Doc :", x)
            print("corpus2/{}.text".format(x))
            f = open("corpus2/{}.text".format(x), "r")
            text = f.read()
            f.close()
            ## SENT TOKENIZE and split up 
            sent_tokenize(text)
            ## split into words
            print("Orginal text :",text)
            tokens = word_tokenize(text)
            print("TOKENIZE:",tokens)
            ## convert to lower case then remove all punctuation except words and space
            tokens = [w.lower() for w in tokens]
            print("lower:",tokens)
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            print("remove punctuation:",stripped)
            ## remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]
            print("not alphabetic:",words)
            ## filter out stop words
            stop_words = set(stopwords)
            words = [w for w in words if not w in stop_words]
            print("filter stop words:",words)
            ## stemming
            stem_word = stem_words(words)
            print("stemming:",stem_word)
            ## lemmetization with pos
            def pos_tagger(nltk_tag):
                if nltk_tag.startswith('J'):
                    return wordnet.ADJ
                elif nltk_tag.startswith('V'):
                    return wordnet.VERB
                elif nltk_tag.startswith('N'):
                    return wordnet.NOUN
                elif nltk_tag.startswith('R'):
                    return wordnet.ADV
                else:
                    return None
            listToStr = ' '.join([str(elem) for elem in words])
            pos_tagged = nltk.pos_tag(nltk.word_tokenize(listToStr))
            print("pos_tagged :",pos_tagged)
            wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
            print("wordnet_tagged :",wordnet_tagged)
            lemmatized_sentence = []
            for word, tag in wordnet_tagged:
                if tag is None:
                    lemmatized_sentence.append(word)
                else:
                    lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            lemmatized_sentence = " ".join(lemmatized_sentence)
            print("lemmatized_sentence :",lemmatized_sentence)
            ### indexing, ranking, matching 
            text2 = proccesedQuery
            res,res2 = tf_idf(text2, words)
            print("search_query_weights ",res)
            print("tfidf_weights_matrix ", res2)
            res3 = cos_similarity(res,res2)
            print("similarity_list ", res3)
            res4 = most_similar(res3)
            print("most_similar ", res4)
            ### matches dates...etc 
            matches = datefinder.find_dates(text)
            # for match in matches:
                # print("match", match)
            spacy_model = spacy.load('en_core_web_sm')
            listToStr2 = ' '.join([str(elem) for elem in words])
            entity_doc = spacy_model(listToStr2)
            print([(entity.text, entity .label_) for entity in entity_doc.ents])
            ### soundex
            sounds_encoding_methods = [soundex, metaphone, nysiis, match_rating_codex]
            report = pd.DataFrame([words]).T
            report.columns = ['word']
            for i in sounds_encoding_methods:
                report[i.__name__] = report['word'].apply(lambda x: i(x))
            # print(report)
        
    ## similarity
    report2 = pd.DataFrame([words]).T
    report2.columns = ['word']
    report.set_index('word', inplace=True)
    report2 = report.copy()
    for sounds_encoding in sounds_encoding_methods:
        report2[sounds_encoding.__name__] = np.nan
        matched_words = []
        for word in words:
            closest_list = []
            for word_2 in words:
                if word != word_2:
                    closest = {}
                    closest['word'] = word_2
                    if (isinstance(report.loc[word, sounds_encoding.__name__], pd.Series) and isinstance(report.loc[word_2, sounds_encoding.__name__], pd.Series)):
                        closest['similarity'] = levenshtein_distance(report.loc[word, sounds_encoding.__name__].values[0], report.loc[word_2, sounds_encoding.__name__].values[0])
                    elif (isinstance(report.loc[word, sounds_encoding.__name__], pd.Series) and not isinstance(report.loc[word_2, sounds_encoding.__name__], pd.Series)):
                        closest['similarity'] = levenshtein_distance(report.loc[word, sounds_encoding.__name__].values[0], report.loc[word_2, sounds_encoding.__name__])
                    elif (not isinstance(report.loc[word, sounds_encoding.__name__], pd.Series) and isinstance(report.loc[word_2, sounds_encoding.__name__], pd.Series)):
                        closest['similarity'] = levenshtein_distance(report.loc[word, sounds_encoding.__name__], report.loc[word_2, sounds_encoding.__name__].values[0])
                    else:
                        closest['similarity'] = levenshtein_distance(report.loc[word, sounds_encoding.__name__],report.loc[word_2, sounds_encoding.__name__])
                    closest_list.append(closest)
            report2.loc[word, sounds_encoding.__name__] = pd.DataFrame(closest_list). \
                sort_values(by='similarity').head(1)['word'].values[0]
    # print(report2)
    print("**************** END ******************")

def searchInFirstDataset():
    submit("first")    
def searchInSecondDataset():
    submit("second")    

window = Tk()
window.title("user input")
label = Label(window,text="Search: ")
label.config(font=("Consolas",30))
label.pack(side=LEFT)
searchInFirstDataset = Button(window,text="search in first dataset",command=searchInFirstDataset)
searchInFirstDataset.pack(side = RIGHT)
searchInSecondDataset = Button(window,text="search in second dataset",command=searchInSecondDataset)
searchInSecondDataset.pack(side = RIGHT)
delete = Button(window,text="delete",command=delete)
delete.pack(side = RIGHT)
backspace = Button(window,text="backspace",command=backspace)
backspace.pack(side = RIGHT)
entry = Entry()
entry.config(font=('areal',22)) 
entry.config(bg='#111111') 
entry.config(fg='#00FF00')
entry.config(width=50)
entry.pack()
window.mainloop()
