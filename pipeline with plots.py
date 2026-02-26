import pandas as pd
import requests 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


from src.DataLoad import DataLoader 
from src.FertilityModel import FertilityModel
from src.Fertility_predictor import FertilityPredictor
from src.Visualization import Visualization

def main(): 
    """
    Main pipeline for predicting female's infertility 
    """
    print("Female Infertility Predictor")

    loader = DataLoader()
    df = loader.load()
    df = loader.clean()

    target_column = "Infertility Prediction"
    X, y = loader.split_features_target(target_column)
    feature_names = X.columns.tolist()

    model = None
    scaler = None 
    last_result = None
    while True: 
        print("MENU")
        print("1 Train Model")
        print("2 Load Model")
        print("3 Predict Patient")
        print("4 Show Patient Report")
        print("5 Exit")

        choice = input("Choice: ").strip()
        if choice == "1":
            print("Training model...")
            X_train, X_test, y_train, y_test, scaler = loader.split_and_scale(X, y)

            model = FertilityModel()
            model.train(X_train, y_train)
            print("Model trained.")

        elif choice == "2":
            model = FertilityModel()
            model.load_model()
            print("Model loaded.")

        elif choice == "3":
            if model is None: 
                print("Train or load model first.")
                continue 

            predictor = FertilityPredictor()
            patient_data = {}
            print("Enter patient data:")
            for feature in feature_names:
                if feature == target_column:
                    continue
                while True: 
                    try: 
                        patient_data[feature] = float(input(f"{feature}: "))
                        break 
                    except ValueError:
                        print("Enter numeric value.")
            
            result = predictor.predict_patient(patient_data)
            last_result = result
            print(result)
        elif choice == "4":
            if model is None:
                print("Train or load model first.")
                continue
            if last_result is None:
                print("Make a prediction first.")
                continue
            viz = Visualization(model)
            viz.plot_patient_report(last_result)
        elif choice == "5":
            print("Exiting system")
            break 
        else:
            print("Invalid choice. Try again.")
if __name__ == "__main__":
    main()
