# -*- coding: utf-8 -*-

import pandas as pd
import re
from nltk.corpus import stopwords
# If you need to, download the stopwords, this is probably only the case when you have not used NLTK before
#nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import multiprocessing as mp
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import csv

##### Define reading, cleaning, and predicting function
def clean_predict_gdelt(read_csv_name, write_csv_name, cv, nb_model, ada_model, gbc_model):
    
    #Open CSV document and create a writer
    with open(write_csv_name,'w', encoding="utf-8", errors="ignore") as writefile:
        writer = csv.writer(writefile, delimiter=',', lineterminator='\n')
        # Read and clean text first by dropping NaN values
        # Then ensure that the text read in actually has text instead of text from a broken link
        text = pd.read_csv(read_csv_name, delimiter = ',', encoding = 'utf8', lineterminator='\n', header = None, low_memory = False)
        text = text.dropna()
        text = text[~text[1].str.contains("403 Forbidden")]
        text = text[~text[1].str.contains("NA")]
        # Re-index everyhting so that you can loop through it properly
        text = text.reset_index(drop = True)
        
        #Clean for text prediction
        for i in range(0, len(text)):
            #Create list to be appended to after cleaning each article
            #This will be used for making the predictions after transforming by the count vectorizer for a BOW model
            corpus = []
            #Regex to eliminate puntuation and symbols
            article = re.sub('[^a-zA-Z]', ' ', str(text[1][i]))
            #Make all letters lowercase so that you don't double count words
            article = article.lower()
            #Split and lemmitize each word
            article = article.split()
            lemm = WordNetLemmatizer()
            article = [lemm.lemmatize(word) for word in article if not word in set(stopwords.words('english'))]
            #Rejoin individual words back into the whole article
            article = ' '.join(article)
            corpus.append(article)
            
            #Predict using cleaned data and term document matrix
            # Transform clean text to term document matrix
            X= cv.transform(corpus).toarray()
            #Predict using pre-loaded / trained Naive Bayes, Adaboost, and Gradient Boosted Classifier models
            y_pred_nb = nb_model.predict(X)
            y_pred_ada_gd = ada_model.predict(X)
            y_pred_gbc_gd = gbc_model.predict(X)
            
            #Ensemble through a hard vote
            hard_vote_gd = y_pred_nb + y_pred_ada_gd + y_pred_gbc_gd
            #If hard vote is positive, predict positive, else predict negative
            if hard_vote_gd > -1:
                hard_vote_gd = 1
            else:
                hard_vote_gd = -1
            
            # If you haev a unique ID for each row ensure that goes here under text[0][i]
            # This is assuming that the unique ID is in the first column of the Dateframe
            unique_row = [text[0][i], hard_vote_gd]
            
            #Write to output CSV
            writer.writerow(unique_row)
            
            #Calculate percentage complete in document
            print ('{:.2%}'.format(i/len(text)), 'complete with document ', read_csv_name)
      

if __name__ == '__main__':
    # Load models and Bog of Words Term Document Matrix
    #Naive Bayes
    nb_model = pickle.load(open('./naive_bayes_model.sav', 'rb'))
    #Adaboost
    ada_model = pickle.load(open('./adaboost_model.sav', 'rb'))
    #Gradient Boosted Classifier
    gbc_model = pickle.load(open('./gbc_model.sav', 'rb'))
    
    #Load Term Document Matrix using pickle loading
    cv = CountVectorizer(vocabulary=pickle.load(open("./cv_bow_feature.pkl", "rb")))
    
    ###########################################################
    #Create list of read and write csv names
    # You can add as many as you want
    read_csv_names = ['./path/to/input_file.csv']
    
    write_csv_names = ['./path/to/output_file.csv']
    
    #This sets the number of precessors to use and then uses the pool.apply function to loop through the predictions
    with mp.Pool(mp.cpu_count()) as pool:
        multiple_results = [pool.apply(clean_predict_gdelt, (read_csv_name, write_csv_name, cv, nb_model, ada_model, gbc_model)) for read_csv_name, write_csv_name in zip(read_csv_names, write_csv_names)]
        
    



