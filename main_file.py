import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.Fertility_model import FertilityModel
from src.Fertility_predictor import FertilityPredictor
from src.Visualizations import Visualization

def main(): 
    """
    Main pipeline for predicting female's infertility 
    """
    predictor = FertilityPredictor()
    viz = Visualization(predictor)
    last_result = None
    model = None
    while True: 
        os.system('clear')
        print(f"{'*' * 60}")
        print("MENU")
        print(f"{'*' * 60}")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Make prediction")
        print("4. Show Patient Report")
        print("5. Show model summary")
        print("6. Exit")

        choice = input("Choice: ").strip()
        if choice == "1":
            predictor.train()
            predictor.save()

        elif choice == "2":
            try:
                predictor.load()
            except Exception as e:
                print(f"Error loading model: {e}")
        
        elif choice == "3":
            if not predictor.is_trained:
                print("Train or load model first.")
                continue
            try:
                predictor.interactive_prediction()
                last_result = predictor.last_result
            except Exception as e:
                print(f"Error during prediction: {e}")

        elif choice == "4":
            if not predictor.is_trained:
                print("Train or load model first.")
                continue
            if last_result is None:
                print("Make a prediction first.")
                continue
            viz.plot_patient_report(last_result)

        elif choice == '5':
            print(str(predictor))
            
        elif choice == "6":
            print("Exiting system.")
            break

        else:
            print("Invalid choice. Try again.")
            
if __name__ == "__main__":
    main()
