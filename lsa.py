"""
@author: imohanty
"""

#Dependencies
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

'''
Accepts dataframe and Performs Latent Semantic Analysis (LSA).
Steps for LSA, 1. use TF-IDF Vectorizer to get Document-Term Matrix.
2. Perform SVD to get information about k-Concepts/Topics in k topics
   (In our case 2 topics - Patent Granted vs Patent Not Granted).
Print the words(terms) in each of the 2 topics
'''

class feature_extraction():
    
    '''
    Constructor: initializes dataframe
    @param df: dataframe
    '''
    def __init__(self,df):
        self.df = df 
    
    '''
    document_term_matrix function:
    Uses TF-IDF Vectorizer to obtain and transform the summary column 
    of the dataframe to a document-term(word) matrix of TF-IDF scores.
    @return dtm: Document-Term Matrix
    @return vectorizer: TF-IDF vectorizer object
    '''
    def document_term_matrix(self):
        vectorizer = TfidfVectorizer(stop_words='english', 
                                     max_features= 2500, # keep top 2500 terms 
                                     max_df = 0.5, 
                                     smooth_idf=True)
        
        dtm = vectorizer.fit_transform(self.df['clean_summary']).toarray()
        
        return dtm,vectorizer 

    '''
    topic_model function:
    Applies SVD to perform dimensionality reduction to M best-features(terms/words)
    and hence, reduce the document-term matrix of TF-IDF scores to M-feature 
    vectors for the k-topics. Here M, may not be known but we consider the
    first 9 best-features(words) and k=2 Topics. Prints the 9-best features for
    the 2 topics: Patent Granted and Patent Not-Granted.
    @param dtm: document-term matrix 
    @param vectorizer: TF-IDF vectorizer object
    '''
    def topic_model(self,dtm,vectorizer):

        # SVD represent documents and terms in vectors 
        svd_model = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=500, random_state=122)
        svd_model.fit(dtm)
        
        #M-best features 
        terms = vectorizer.get_feature_names()
        
        #Display the 9-best Features for the 2-topics
        for i, comp in enumerate(svd_model.components_):
            terms_comp = zip(terms, comp)
            #take 9-best features
            sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:9] 
            print("Topic "+str(i)+": \n")
            for t in sorted_terms:
                print(t[0])
            print(" ")   
        
