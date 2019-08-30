"""
@author: imohanty
"""

#Dependencies
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

'''
Classifies Patents as Granted or Not-Granted using Different Machine-Learning Algorithms
and Evaluates these Classifiers based on metrics such as Confusion Matrix, Accuracy, Precision,
Recall and F1-score.
'''
class classify():
    
    '''
    Constructor: initializes train-test data and labels
    @param X_train: training data
    @param X_test: test data
    @param y_train: training labels
    @param y_test: test labels
    '''
    def __init__(self,X_train,X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
    
    '''
    make_classification function: 
    Classifies patents as granted(1) or Not-granted(0) using different ML-Algorithms 
    and performs evalauation of these models using metrics such as Confusion Matrix, Accuracy, 
    Precision, Recall and F1-score. 
    '''    
    def make_classification(self):
        
        names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "LGBM", "XGB", "Logistic Regression"]
    
        classifiers = [
                        KNeighborsClassifier(5),
                        SVC(kernel="linear"),
                        SVC(kernel="rbf"),
                        DecisionTreeClassifier(max_depth=5),
                        RandomForestClassifier(n_estimators=1000, random_state=0,class_weight="balanced"),
                        MLPClassifier(alpha=0.01, max_iter=100),
                        AdaBoostClassifier(),
                        lgb.LGBMClassifier(max_depth=3, random_state = 314, num_leaves = 31 ),
                        xgb.XGBClassifier(max_depth=5, random_state = 314 ),
                        LogisticRegression(class_weight="balanced")
                      ]
        
        #Prediction(classification) and Evaluation
        for name, clf in zip(names, classifiers):
            clf.fit(self.X_train, self.y_train)
            predictions = clf.predict(self.X_test)
            print("-----------------------------------------------")
            print(name)
            print("-----------------------------------------------")
            print(" ")
            print(confusion_matrix(self.y_test,predictions))
            print(classification_report(self.y_test,predictions))
            print("accuracy: ",accuracy_score(self.y_test, predictions))
            print(" ")
            print("-----------------------------------------------")
            print(" ")
    
        
        
        
    