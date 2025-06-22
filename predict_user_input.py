import pickle
import pandas as pd

def predict_disease_from_symptoms(symptom_text: str):
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    with open('ensemble_model.pkl', 'rb') as f:
        model = pickle.load(f)

    X = tfidf.transform([symptom_text])
    X_df = pd.DataFrame(X.toarray(), columns=tfidf.get_feature_names_out())
    
    # Align columns to match model
    with open('selected_columns.pkl', 'rb') as f:
        selected_cols = pickle.load(f)

    # Fill missing features with 0
    for col in selected_cols:
        if col not in X_df.columns:
            X_df[col] = 0
    X_df = X_df[selected_cols]

    pred_encoded = model.predict(X_df)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    return pred_label
