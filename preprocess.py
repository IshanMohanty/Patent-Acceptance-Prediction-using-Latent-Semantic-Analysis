"""
@author: imohanty
"""

#Dependencies
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

'''
performs text preprocessing by cleaning and preparing text for semantic
analysis.
'''
class text_preprocess():
    
    '''
    Constructor: initializes dataframe
    @param df: dataframe
    '''
    def __init__(self,df):
        self.df = df 
    
    '''
    text_cleaning function: 
    remove non-alphabets, short words and transform to lowercase.
    '''    
    def text_cleaning(self):
        # remove everything except alphabets`
        self.df['clean_summary'] = self.df['summary'].str.replace("[^a-zA-Z#]", " ")
        # remove short words
        self.df['clean_summary'] = self.df['summary'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
        # make all text to lowercase
        self.df['clean_summary'] = self.df['clean_summary'].apply(lambda x: x.lower())
    
    '''
    prepare_text function: 
    transform text by removing stop words, stemming and lemmatizing tokens and
    detokenize to form sentences.
    @return: final prepocessed summary column in the dataframe 
    '''    
    def prepare_text(self):
        stop_words = stopwords.words('english')
        
        # tokenization
        tokenized_doc = self.df['clean_summary'].apply(lambda x: x.split())
        # remove stop-words
        tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
        
        #perform stemming 
        port = PorterStemmer()
        tokenized_doc = tokenized_doc.apply(lambda x: [port.stem(item) for item in x ])
        
        #perform lemmatization
        wnl = WordNetLemmatizer()
        tokenized_doc = tokenized_doc.apply(lambda x: [wnl.lemmatize(item) for item in x ])
        
        # de-tokenization
        detokenized_doc = []
        for i in tokenized_doc:
            t = ' '.join(i)
            detokenized_doc.append(t)
        self.df['clean_summary'] = detokenized_doc
        
        return self.df 
    
        