"""
@author: imohanty
"""
#Dependencies 
import warnings
warnings.filterwarnings("ignore")

from file_convert import json_to_df
from preprocess import text_preprocess
from lsa import feature_extraction
from classification import classify
from sklearn.model_selection import train_test_split

'''
Main function: 
1. Reads Json file, Processes it and converts to dataframe.
2. Perform Text cleaning ( remove spaces, punctuations(etc), short words, lower-casing) 
3. Does Text Preparation ( stop-words(remove), stemming and lemmatization )
4. Apply Latent Sematic Analysis(LSA) for feature extraction
5. Classification task based on the features from LSA into 2 classes,
   Patent Granted and Patent Not-Granted. 
'''
if __name__ == "__main__":
    
    #convert json to dataframe
    df = json_to_df('uspto.json')
    
    #preprocess text from the summary column of the dataframe
    tp = text_preprocess(df)
    tp.text_cleaning()
    df = tp.prepare_text()
    
    #Latent Semantic Analysis for feature extraction
    lsa = feature_extraction(df)
    dtm, vectorizer = lsa.document_term_matrix()
    lsa.topic_model(dtm,vectorizer)
    
    #Train Model for classification
    X_train, X_test, y_train, y_test = train_test_split(dtm, df['Decision'].astype('int'), test_size=0.20, random_state=0)
    clf = classify(X_train, X_test, y_train, y_test)
    clf.make_classification()
    
    
        
    
    
    