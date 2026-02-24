import os
from src.Fertility_predictor import FertilityPredictor
from src.Visualizations import Visualization

def main():
    predictor = FertilityPredictor()
    viz = Visualization(predictor)

    while True:
        print(f"\n{'=' * 60}")
        print("FERTILITY PREDICTION SYSTEM")
        print(f"{'=' * 60}")
        print("  1. Train new model")
        print("  2. Load existing model")
        print("  3. Make prediction (interactive)")
        print("  4. Show feature importances")
        print("  5. Show model summary")
        print("  6. Show patient report (plots)")
        print("  7. Exit")
        print(f"{'=' * 60}")

        choice = input("Choice (1-7): ").strip()

        if choice == '1':
            target = input("Target column [Infertility Prediction]: ").strip()
            target = target or 'Infertility Prediction'
            try:
                predictor.train(target_column=target)
                if input("Save model? (y/n): ").strip().lower() == 'y':
                    predictor.save()
            except Exception as e:
                print(f"\n❌ Error: {e}")

        elif choice == '2':
            if not os.path.exists('fertility_model.pkl'):
                print("\n❌ No saved model found. Train first (option 1).")
            else:
                try:
                    predictor.load()
                except Exception as e:
                    print(f"\n❌ Error loading: {e}")

        elif choice == '3':
            if not predictor.is_trained:
                print("\n  Train or load a model first.")
            else:
                try:
                    predictor.interactive_prediction()
                except Exception as e:
                    print(f"\n❌ Error: {e}")

        elif choice == '4':
            if not predictor.is_trained:
                print("\n  Train or load a model first.")
            else:
                print(predictor.feature_importance())

        elif choice == '5':
            print(str(predictor))

        elif choice == '6':
            if not predictor.is_trained:
                print("\n  Train or load a model first.")
            elif predictor.last_result is None:
                print("\n Make a prediction first (option 3).")
            else:
                viz.plot_patient_report(predictor.last_result)

        elif choice == '7':
            print("\nGoodbye!")
            break

        else:
            print("\n❌ Invalid choice — please enter 1–7.")


if __name__ == "__main__":
    main()
