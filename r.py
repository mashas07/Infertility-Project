"""
Complete Fertility Prediction System - WORKING VERSION
All bugs fixed, ready to use.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import warnings

warnings.filterwarnings('ignore')

# Your CSV file location
DEFAULT_CSV_PATH = "/Users/mashasobko/Downloads/Female_infertility.csv"


class FertilityPredictor:
    """Complete fertility prediction system."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}  # Store encoders for each categorical feature
        self.target_encoder = LabelEncoder()
        self.feature_names = None
        self.original_features = None
        self.feature_types = {}
        self.target_classes = None
        self.is_trained = False

    def train(self, csv_path=DEFAULT_CSV_PATH, target_column='Infertility Prediction'):
        """Train the model from scratch."""
        print(f"\n{'=' * 60}")
        print("TRAINING NEW MODEL")
        print(f"{'=' * 60}")
        print(f"Data source: {csv_path}")
        print(f"Target: {target_column}")

        # Load data
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        print(f"\nData loaded: {df.shape}")

        # Validate target
        if target_column not in df.columns:
            raise ValueError(f"Target '{target_column}' not found. Available: {list(df.columns)}")

        # Clean data
        print("Cleaning data...")
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)

        df = df.drop_duplicates()

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Store original features
        self.original_features = X.columns.tolist()

        print(f"\n{'=' * 60}")
        print("FEATURES TO BE USED:")
        print(f"{'=' * 60}")
        for i, col in enumerate(self.original_features, 1):
            dtype = 'numeric' if X[col].dtype in ['int64', 'float64'] else 'categorical'
            self.feature_types[col] = dtype
            print(f"{i:2d}. {col:40s} ({dtype})")
        print(f"{'=' * 60}\n")

        # Encode categorical features
        X_encoded = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                self.label_encoders[col] = LabelEncoder()
                X_encoded[col] = self.label_encoders[col].fit_transform(X[col].astype(str))

        self.feature_names = X_encoded.columns.tolist()

        # Encode target
        if y.dtype == 'object':
            y_encoded = self.target_encoder.fit_transform(y)
            self.target_classes = self.target_encoder.classes_
            print(f"Target classes: {list(self.target_classes)}\n")
        else:
            y_encoded = y
            self.target_classes = None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}\n")

        # Scale and train
        print("Training Random Forest...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )

        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{'=' * 60}")
        print("MODEL EVALUATION")
        print(f"{'=' * 60}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=self.target_classes if self.target_classes is not None else None))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"{'=' * 60}\n")

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)

        print("TOP 10 MOST IMPORTANT FEATURES:")
        print(f"{'=' * 60}")
        for _, row in importance.iterrows():
            bar = "█" * int(row['importance'] * 50)
            print(f"{row['feature']:40s} {bar} {row['importance']:.4f}")
        print(f"{'=' * 60}\n")

        return accuracy

    def predict_single(self, user_data):
        """Make prediction on single user data."""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        # Validate features
        missing = set(self.original_features) - set(user_data.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Create dataframe
        user_df = pd.DataFrame([user_data])[self.original_features]

        # Encode categorical features
        for col in user_df.columns:
            if col in self.label_encoders:
                try:
                    user_df[col] = self.label_encoders[col].transform(user_df[col].astype(str))
                except ValueError as e:
                    # Unknown category - use most frequent
                    print(f"Warning: Unknown value for {col}, using default")
                    user_df[col] = 0

        # Scale
        user_scaled = self.scaler.transform(user_df)

        # Predict
        prediction = self.model.predict(user_scaled)[0]
        probabilities = self.model.predict_proba(user_scaled)[0]

        # Decode
        if self.target_classes is not None:
            predicted_class = self.target_encoder.inverse_transform([prediction])[0]
        else:
            predicted_class = prediction

        confidence = max(probabilities) * 100

        result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {}
        }

        if self.target_classes is not None:
            for i, cls in enumerate(self.target_classes):
                result['probabilities'][cls] = probabilities[i] * 100

        return result

    def interactive_prediction(self):
        """Get user input and make prediction."""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")

        print(f"\n{'=' * 60}")
        print("FERTILITY PREDICTION - DATA INPUT")
        print(f"{'=' * 60}")
        print("Please provide the following information:\n")

        user_data = {}

        for feature in self.original_features:
            while True:
                try:
                    value = input(f"{feature}: ").strip()
                    if not value:
                        print(f"  ⚠️  {feature} is required!")
                        continue

                    # Convert to appropriate type
                    if self.feature_types[feature] == 'numeric':
                        user_data[feature] = float(value)
                    else:
                        user_data[feature] = value
                    break
                except ValueError:
                    print(f"  ⚠️  Please enter a valid number for {feature}")

        print("\nProcessing...")

        try:
            result = self.predict_single(user_data)

            print(f"\n{'=' * 60}")
            print("PREDICTION RESULTS")
            print(f"{'=' * 60}")
            print(f"\n✓ Prediction: {result['prediction']}")
            print(f"✓ Confidence: {result['confidence']:.2f}%")

            if result['probabilities']:
                print(f"\nProbability Breakdown:")
                for cls, prob in result['probabilities'].items():
                    bar = "█" * int(prob / 2)
                    print(f"  {cls:20s} {bar} {prob:.2f}%")

            print(f"\n{'=' * 60}")
            print("⚠️  DISCLAIMER: This is a prediction, not a diagnosis.")
            print("Please consult healthcare professionals.")
            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def save(self, model_path='fertility_model.pkl', components_path='fertility_components.pkl'):
        """Save model and components."""
        if not self.is_trained:
            raise ValueError("Train model first!")

        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save everything else
        components = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'target_encoder': self.target_encoder,
            'feature_names': self.feature_names,
            'original_features': self.original_features,
            'feature_types': self.feature_types,
            'target_classes': self.target_classes
        }

        with open(components_path, 'wb') as f:
            pickle.dump(components, f)

        print(f"✓ Saved: {model_path}")
        print(f"✓ Saved: {components_path}")

    def load(self, model_path='fertility_model.pkl', components_path='fertility_components.pkl'):
        """Load model and components."""
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load components
        with open(components_path, 'rb') as f:
            components = pickle.load(f)

        self.scaler = components['scaler']
        self.label_encoders = components['label_encoders']
        self.target_encoder = components['target_encoder']
        self.feature_names = components['feature_names']
        self.original_features = components['original_features']
        self.feature_types = components['feature_types']
        self.target_classes = components['target_classes']

        self.is_trained = True

        print(f"✓ Loaded: {model_path}")
        print(f"✓ Loaded: {components_path}")
        print(f"\nModel trained on {len(self.original_features)} features")


def main():
    """Main menu."""
    predictor = FertilityPredictor()

    while True:
        print(f"\n{'=' * 60}")
        print("FERTILITY PREDICTION SYSTEM")
        print(f"{'=' * 60}")
        print("1. Train new model")
        print("2. Load existing model")
        print("3. Make prediction")
        print("4. Exit")
        print(f"{'=' * 60}")

        choice = input("\nChoice (1-4): ").strip()

        if choice == '1':
            try:
                target = input("\nTarget column (default: 'Ovulation Disorders'): ").strip()
                target = target if target else 'Ovulation Disorders'

                predictor.train(target_column=target)

                save = input("\nSave model? (y/n): ").strip().lower()
                if save == 'y':
                    predictor.save()
                    print("✓ Model saved!\n")

            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()

        elif choice == '2':
            try:
                if not os.path.exists('fertility_model.pkl'):
                    print("\n❌ No saved model found. Train first (option 1).")
                    continue

                predictor.load()

            except Exception as e:
                print(f"\n❌ Error loading: {str(e)}")
                print("Try deleting .pkl files and retraining.")

        elif choice == '3':
            if not predictor.is_trained:
                print("\n⚠️  Train or load a model first!")
            else:
                try:
                    predictor.interactive_prediction()
                except Exception as e:
                    print(f"\n❌ Error: {str(e)}")
                    import traceback
                    traceback.print_exc()

        elif choice == '4':
            print("\nGoodbye!")
            break

        else:
            print("\n❌ Invalid choice")


if __name__ == "__main__":
    main()