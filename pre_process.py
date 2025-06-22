import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from merge_datasets import merge_disease_symptoms
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
import pickle

def pre_process():
    df = merge_disease_symptoms('datasets/df.csv', 'datasets/df2.csv', 'datasets/df3.csv')
    print("dataset size=", df.shape)
    print("Null Values in DF=", df.isnull().sum())

    df['PatientID'] = ['P' + str(i) for i in range(1, len(df) + 1)]

    grouped = df.groupby('PatientID').agg({
        'Symptom': lambda x: ' '.join(x),
        'Disease': 'first'
    }).reset_index()

    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = tfidf.fit_transform(grouped['Symptom'])

    tfidf_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())
    tfidf_df['Disease'] = grouped['Disease']
    tfidf_df['PatientID'] = grouped['PatientID']  # only if you wanna keep it temporarily

    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    tfidf_df = select_imp_features(tfidf_df)
    print("dataset size after tf-idf=", tfidf_df.shape)
    print(tfidf_df.head(4))
    return tfidf_df


def select_imp_features(df):
    le = LabelEncoder()
    df['Disease'] = le.fit_transform(df['Disease'])

    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    selector = SelectKBest(score_func=chi2, k=100)
    X_selected = selector.fit_transform(df.drop(columns=['Disease', 'PatientID'], errors='ignore'), df['Disease'])

    selected_cols = df.drop(columns=['Disease', 'PatientID'], errors='ignore').columns[selector.get_support()]
    final_df = pd.DataFrame(X_selected, columns=selected_cols)
    final_df['Disease'] = df['Disease']
    with open('selected_columns.pkl', 'wb') as f:
     pickle.dump(list(final_df.columns[:-1]), f)  # exclude Disease


    return final_df

