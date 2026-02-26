import pandas as pd
from datasets import load_dataset
from src.preprocessing import encode_categorical, scale_features
from src.model import train_random_forest
from src.visualization import (
    plot_patient_distribution,
    plot_importance_and_values,
    plot_similar_patients,
    plot_patient_vs_average
)

def main():
    dataset = load_dataset("mstz/fertility", "fertility")["train"]
    df = pd.DataFrame(dataset)

    df_encoded, le_dict = encode_categorical(df)
    X = df_encoded.drop('has_fertility_issues', axis=1)
    y = df_encoded['has_fertility_issues']
    X_scaled, scaler = scale_features(X)

    model = train_random_forest(X_scaled, y)

    patient_index = 0
    patient_scaled = X_scaled[patient_index].reshape(1, -1)
    patient_raw = X.iloc[patient_index].to_dict()

    fig1 = plot_patient_distribution(df['age_at_time_of_sampling'], patient_raw['age_at_time_of_sampling'], 'Age')
    fig2 = plot_importance_and_values(model, X.columns.tolist(), patient_raw)
    fig3 = plot_similar_patients(X_scaled, y, patient_scaled)
    fig4 = plot_patient_vs_average(df, patient_raw)

if __name__ == "__main__":
    main()
