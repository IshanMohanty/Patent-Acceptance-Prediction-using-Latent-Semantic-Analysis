"""
@author: imohanty
"""

Install Dependencies:

python3
pandas 
sklearn
lightgbm
xgboost
nltk

Program modules:

1. file_convert.py: json to dataframe
2. preprocess.py: preprocess and prepare text
3. lsa.py: latent semantic analysis for feature extraction
4. classification.py: classify patent granted or not granted 
5. main.py: driver of the application

json file data:

put the json file "uspto.json" in the same directory level / Path as the Program modules

Run:
main.py

Display Results:
1. 2 topic models with top 9 features/words
2. Classifier Performance Report - Confusion Matrix, Precision, Recall and F1-score.





