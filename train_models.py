import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
from pre_process import pre_process  

def train_models():
    df = pre_process()
    X = df.drop(columns=['Disease'])
    y = df['Disease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    nb = MultinomialNB()
    svm = LinearSVC()

    ensemble = VotingClassifier(estimators=[
        ('rf', rf),
        ('nb', nb),
        ('svm', svm)
    ], voting='hard')

    ensemble.fit(X_train, y_train)

    with open('ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)

    print("Model Accuracy =", ensemble.score(X_test, y_test))


