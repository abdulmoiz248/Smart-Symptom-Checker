import pandas as pd

def merge_disease_symptoms(df1_path, df2_path, df3_path):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    df3 = pd.read_csv(df3_path)

    df1_melted = df1.melt(id_vars=['diseases'], var_name='Symptom', value_name='Present')
    df1_melted = df1_melted[df1_melted['Present'] == 1]
    df1_melted = df1_melted.rename(columns={'diseases': 'Disease'})
    df1_clean = df1_melted[['Disease', 'Symptom']]

    df2['Symptoms'] = df2['Symptoms'].str.split(', ')
    df2_exploded = df2.explode('Symptoms')
    df2_exploded = df2_exploded.rename(columns={
        'Symptoms': 'Symptom',
        'Predicted Disease': 'Disease'
    })
 
    df2_clean = df2_exploded[['Disease', 'Symptom']]

    symptom_cols = [col for col in df3.columns if col.startswith('Symptom_')]
    df3_melted = df3.melt(id_vars=['Disease'], value_vars=symptom_cols, var_name='Symptom_num', value_name='Symptom')
    df3_melted = df3_melted.dropna(subset=['Symptom'])
    df3_clean = df3_melted[['Disease', 'Symptom']]

    for df in [df1_clean, df2_clean, df3_clean]:
        df['Symptom'] = df['Symptom'].str.strip().str.lower().str.replace(' ', '_')

    merged_df = pd.concat([df1_clean, df2_clean, df3_clean], ignore_index=True)

    return merged_df




