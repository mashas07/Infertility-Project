"""
Fertility Prediction System - Refactored OOP Version
Classes: DataLoader, FertilityModel, PatientPredictor, Visualization, Main
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
import kagglehub
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings('ignore')

class DataLoader:
    """Responsible for loading, validating, and cleaning CSV data."""

    def __init__(self):
        self._df: pd.DataFrame | None = None

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Data not loaded yet. Call load() first.")
        return self._df

    @property
    def shape(self):
        return self.dataframe.shape

    def __repr__(self):
        loaded = f"loaded {self._df.shape}" if self._df is not None else "not loaded"
        return f"DataLoader(kaggle, data={loaded})"

    def __str__(self):
        if self._df is None:
            return "DataLoader → kaggle (not loaded)"
        return f"DataLoader → kaggle | {self._df.shape[0]} rows × {self._df.shape[1]} cols"

    def __len__(self):
        return len(self.dataframe)

    def load(self) -> pd.DataFrame:
        '''Load CSV and strip column whitespace.'''
        path = kagglehub.dataset_download("fida5073/female-infertility")
        csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
        self._df = pd.read_csv(os.path.join(path, csv_file))
        self._df.columns = self._df.columns.str.strip()
        print(f"Loaded: {self._df.shape[0]} rows × {self._df.shape[1]} columns")
        return self._df

    def clean(self) -> pd.DataFrame:
        '''Drop columns with many missing values and duplicates.'''
        df = self.dataframe
        self._df.drop(columns=['Patient ID'], inplace=True, errors='ignore')

        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        before = len(df)
        self._df = df.drop_duplicates().reset_index(drop=True)
        dropped = before - len(self._df)
        if dropped:
            print(f"Removed {dropped} duplicate rows")
        return self._df

    def split_features_target(self, target_column: str):
        '''Return (X, y) split from the loaded dataframe.'''
        df = self.dataframe
        if target_column not in df.columns:
            raise ValueError(
                f"Target '{target_column}' not found. Available: {list(df.columns)}"
            )
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y

class FertilityModel:
    """
    Trains and persists a Random Forest fertility classifier.
    Composes a DataLoader for data access.
    """

    def __init__(self):
        # composition
        self._loader = DataLoader()

        self._model: RandomForestClassifier | None = None
        self._scaler = StandardScaler()
        self._label_encoders: dict[str, LabelEncoder] = {}
        self._target_encoder = LabelEncoder()
        self._feature_names: list[str] = []
        self._original_features: list[str] = []
        self._feature_types: dict[str, str] = {}
        self._target_classes = None
        self._accuracy: float | None = None
        self._is_trained: bool = False

    # ── properties ──────────────────────────────

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def accuracy(self) -> float | None:
        return self._accuracy

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    @property
    def original_features(self) -> list[str]:
        return self._original_features

    @property
    def loader(self) -> DataLoader:
        return self._loader

    # ── magic methods ────────────────────────────

    def __repr__(self):
        status = f"trained, accuracy={self._accuracy:.4f}" if self._is_trained else "untrained"
        return f"FertilityModel({status}, features={len(self._original_features)})"

    def __str__(self):
        if not self._is_trained:
            return "FertilityModel [untrained]"
        cls = list(self._target_classes) if self._target_classes is not None else "numeric"
        return (
            f"FertilityModel\n"
            f"  Accuracy : {self._accuracy * 100:.2f}%\n"
            f"  Features : {len(self._original_features)}\n"
            f"  Classes  : {cls}"
        )

    def __eq__(self, other):
        if not isinstance(other, FertilityModel):
            return NotImplemented
        return (
            self._is_trained == other._is_trained
            and self._accuracy == other._accuracy
            and self._original_features == other._original_features
        )

    # ── public methods ───────────────────────────

    def train(self, target_column: str = 'Ovulation Disorders') -> float:
        """Load data, encode, train, and evaluate the model."""
        print(f"\n{'=' * 60}")
        print("TRAINING NEW MODEL")
        print(f"{'=' * 60}")

        # Load & clean via composed DataLoader
        self._loader.load()
        self._loader.clean()
        X, y = self._loader.split_features_target(target_column)

        self._original_features = X.columns.tolist()

        print(f"\nFEATURES ({len(self._original_features)}):")
        for i, col in enumerate(self._original_features, 1):
            dtype = 'numeric' if X[col].dtype in ['int64', 'float64'] else 'categorical'
            self._feature_types[col] = dtype
            print(f"  {i:2d}. {col:40s} ({dtype})")

        # Encode categoricals
        X_encoded = self._encode_features(X, fit=True)
        self._feature_names = X_encoded.columns.tolist()

        # Encode target
        if y.dtype == 'object':
            y_encoded = self._target_encoder.fit_transform(y)
            self._target_classes = self._target_encoder.classes_
            print(f"\nTarget classes: {list(self._target_classes)}")
        else:
            y_encoded = y.values
            self._target_classes = None

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        print(f"\nTrain: {X_train.shape}  |  Test: {X_test.shape}")

        # Scale
        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s = self._scaler.transform(X_test)

        # Fit
        print("Training Random Forest…")
        self._model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10, min_samples_split=5
        )
        self._model.fit(X_train_s, y_train)
        self._is_trained = True

        # Evaluate
        y_pred = self._model.predict(X_test_s)
        self._accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{'=' * 60}")
        print("MODEL EVALUATION")
        print(f"{'=' * 60}")
        print(f"Accuracy: {self._accuracy * 100:.2f}%\n")
        print(classification_report(
            y_test, y_pred,
            target_names=self._target_classes if self._target_classes is not None else None
        ))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"{'=' * 60}")

        return self._accuracy

    def predict_encoded(self, user_df: pd.DataFrame):
        """Encode, scale, and return (prediction, probabilities) for a prepared DataFrame."""
        if not self._is_trained:
            raise RuntimeError("Train the model before predicting.")
        X_enc = self._encode_features(user_df[self._original_features], fit=False)
        X_scaled = self._scaler.transform(X_enc)
        prediction = self._model.predict(X_scaled)[0]
        probabilities = self._model.predict_proba(X_scaled)[0]
        return prediction, probabilities

    def decode_prediction(self, prediction, probabilities):
        """Convert encoded prediction back to human-readable form."""
        if self._target_classes is not None:
            predicted_class = self._target_encoder.inverse_transform([prediction])[0]
        else:
            predicted_class = prediction
        confidence = float(max(probabilities)) * 100
        prob_dict = {}
        if self._target_classes is not None:
            for i, cls in enumerate(self._target_classes):
                prob_dict[cls] = probabilities[i] * 100
        return predicted_class, confidence, prob_dict

    def feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Train the model first.")
        return (
            pd.DataFrame({'feature': self._feature_names,
                          'importance': self._model.feature_importances_})
            .sort_values('importance', ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

    def save(self, model_path: str = 'fertility_model.pkl',
             components_path: str = 'fertility_components.pkl'):
        if not self._is_trained:
            raise RuntimeError("Train the model before saving.")
        with open(model_path, 'wb') as f:
            pickle.dump(self._model, f)
        components = {
            'scaler': self._scaler,
            'label_encoders': self._label_encoders,
            'target_encoder': self._target_encoder,
            'feature_names': self._feature_names,
            'original_features': self._original_features,
            'feature_types': self._feature_types,
            'target_classes': self._target_classes,
            'accuracy': self._accuracy,
        }
        with open(components_path, 'wb') as f:
            pickle.dump(components, f)
        print(f"  ✓ Saved model      → {model_path}")
        print(f"  ✓ Saved components → {components_path}")

    def load(self, model_path: str = 'fertility_model.pkl',
             components_path: str = 'fertility_components.pkl'):
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
        with open(components_path, 'rb') as f:
            c = pickle.load(f)
        self._scaler = c['scaler']
        self._label_encoders = c['label_encoders']
        self._target_encoder = c['target_encoder']
        self._feature_names = c['feature_names']
        self._original_features = c['original_features']
        self._feature_types = c['feature_types']
        self._target_classes = c['target_classes']
        self._accuracy = c.get('accuracy')
        self._is_trained = True
        print(f"  ✓ Loaded model      ← {model_path}")
        print(f"  ✓ Loaded components ← {components_path}")
        print(f"  Features: {len(self._original_features)}")

    # ── private helpers ──────────────────────────

    def _encode_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X_enc = X.copy()
        for col in X_enc.columns:
            if X_enc[col].dtype == 'object':
                if fit:
                    self._label_encoders[col] = LabelEncoder()
                    X_enc[col] = self._label_encoders[col].fit_transform(
                        X_enc[col].astype(str)
                    )
                else:
                    if col in self._label_encoders:
                        try:
                            X_enc[col] = self._label_encoders[col].transform(
                                X_enc[col].astype(str)
                            )
                        except ValueError:
                            print(f"  ⚠  Unknown category for '{col}', defaulting to 0")
                            X_enc[col] = 0
        return X_enc


# ─────────────────────────────────────────────
# 3. PatientPredictor  (inherits FertilityModel)
# ─────────────────────────────────────────────

class PatientPredictor(FertilityModel):
    """
    Extends FertilityModel with patient-facing prediction helpers.
    Adds input validation, single-patient prediction, and interactive CLI.
    """

    def __init__(self):
        super().__init__()
        self._last_result: dict | None = None

    # ── properties ──────────────────────────────

    @property
    def last_result(self) -> dict | None:
        return self._last_result

    # ── magic methods ────────────────────────────

    def __repr__(self):
        base = super().__repr__().replace("FertilityModel", "PatientPredictor")
        last = f", last_prediction='{self._last_result['prediction']}'" \
            if self._last_result else ""
        return base.rstrip(')') + last + ')'

    def __str__(self):
        base = super().__str__().replace("FertilityModel", "PatientPredictor")
        if self._last_result:
            base += f"\n  Last prediction : {self._last_result['prediction']} " \
                    f"({self._last_result['confidence']:.1f}%)"
        return base

    # ── public methods ───────────────────────────

    def predict_patient(self, patient_data: dict) -> dict:
        """Validate patient dict and return prediction result."""
        if not self.is_trained:
            raise RuntimeError("Train or load a model first.")

        # Validate all required features present
        missing = set(self._original_features) - set(patient_data.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")

        # Validate numeric fields
        for feat in self._original_features:
            if self._feature_types.get(feat) == 'numeric':
                try:
                    patient_data[feat] = float(patient_data[feat])
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Feature '{feat}' must be numeric, got: {patient_data[feat]!r}"
                    )

        user_df = pd.DataFrame([patient_data])
        raw_pred, probs = self.predict_encoded(user_df)
        predicted_class, confidence, prob_dict = self.decode_prediction(raw_pred, probs)

        self._last_result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': prob_dict,
            'patient_data': patient_data,
        }
        return self._last_result

    def interactive_prediction(self):
        """Collect patient data interactively from the CLI and display results."""
        if not self.is_trained:
            raise RuntimeError("Train or load a model first.")

        print(f"\n{'=' * 60}")
        print("PATIENT DATA INPUT")
        print(f"{'=' * 60}")

        patient_data = {}
        for feature in self._original_features:
            while True:
                raw = input(f"  {feature}: ").strip()
                if not raw:
                    print(f"    ⚠  '{feature}' is required.")
                    continue
                if self._feature_types[feature] == 'numeric':
                    try:
                        patient_data[feature] = float(raw)
                        break
                    except ValueError:
                        print(f"    ⚠  Please enter a valid number.")
                else:
                    patient_data[feature] = raw
                    break

        print("\n  Processing…")
        result = self.predict_patient(patient_data)
        self._display_result(result)

    # ── private helpers ──────────────────────────

    @staticmethod
    def _display_result(result: dict):
        print(f"\n{'=' * 60}")
        print("PREDICTION RESULTS")
        print(f"{'=' * 60}")
        print(f"  ✓ Prediction : {result['prediction']}")
        print(f"  ✓ Confidence : {result['confidence']:.2f}%")
        if result['probabilities']:
            print("\n  Probability breakdown:")
            for cls, prob in result['probabilities'].items():
                bar = "█" * int(prob / 2)
                print(f"    {cls:20s} {bar} {prob:.1f}%")
        print(f"\n{'=' * 60}")
        print("  ⚠  This is a prediction, not a medical diagnosis.")
        print("     Please consult a qualified healthcare professional.")
        print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────
# 4. Visualization  (composes FertilityModel)
# ─────────────────────────────────────────────

class Visualization:
    """
    ASCII / text-based visualization helpers.
    Composes a FertilityModel (or subclass) to access model internals.
    """

    BAR_WIDTH = 40

    def __init__(self, model: FertilityModel):
        if not isinstance(model, FertilityModel):
            raise TypeError("model must be a FertilityModel instance.")
        self._model = model  # composition

    # ── magic methods ────────────────────────────

    def __repr__(self):
        return f"Visualization(model={self._model!r})"

    def __str__(self):
        return f"Visualization → bound to {self._model.__class__.__name__}"

    # ── public methods ───────────────────────────

    def show_feature_importance(self, top_n: int = 10):
        """Print a horizontal bar chart of feature importances."""
        df = self._model.feature_importance(top_n)
        print(f"\n{'=' * 60}")
        print(f"TOP {top_n} FEATURE IMPORTANCES")
        print(f"{'=' * 60}")
        max_imp = df['importance'].max()
        for _, row in df.iterrows():
            bar_len = int((row['importance'] / max_imp) * self.BAR_WIDTH)
            bar = "█" * bar_len
            print(f"  {row['feature']:38s} {bar} {row['importance']:.4f}")
        print(f"{'=' * 60}\n")

    def show_probability_breakdown(self, result: dict):
        """Render probability bars for a prediction result dict."""
        if not result.get('probabilities'):
            print("  (no probability breakdown available)")
            return
        print(f"\n  Probability breakdown:")
        for cls, prob in result['probabilities'].items():
            bar = "█" * int(prob / 2)
            print(f"    {cls:22s} {bar} {prob:.1f}%")

    def show_model_summary(self):
        """Print a structured summary of the bound model."""
        print(str(self._model))

    def show_data_summary(self):
        """Print basic stats about the loaded dataset."""
        try:
            df = self._model.loader.dataframe
        except (AttributeError, RuntimeError) as exc:
            print(f"  ⚠  {exc}")
            return
        print(f"\n{'=' * 60}")
        print("DATASET SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Rows    : {df.shape[0]}")
        print(f"  Columns : {df.shape[1]}")
        nulls = df.isnull().sum().sum()
        print(f"  Nulls   : {nulls}")
        print(f"\n  Column dtypes:")
        for col, dtype in df.dtypes.items():
            print(f"    {col:40s} {str(dtype)}")
        print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────
# 5. Main / CLI
# ─────────────────────────────────────────────

def main():
    predictor = PatientPredictor()
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
        print("  6. Exit")
        print(f"{'=' * 60}")

        choice = input("Choice (1-6): ").strip()

        if choice == '1':
            target = input("Target column [Ovulation Disorders]: ").strip()
            target = target or 'Ovulation Disorders'
            try:
                predictor.train(target_column=target)
                viz.show_feature_importance()
                if input("Save model? (y/n): ").strip().lower() == 'y':
                    predictor.save()
            except Exception as e:
                print(f"\n❌ Error: {e}")

        elif choice == '2':
            model_path = 'fertility_model.pkl'
            comp_path = 'fertility_components.pkl'
            if not os.path.exists(model_path):
                print("\n❌ No saved model found. Train first (option 1).")
            else:
                try:
                    predictor.load(model_path, comp_path)
                except Exception as e:
                    print(f"\n❌ Error loading: {e}")

        elif choice == '3':
            if not predictor.is_trained:
                print("\n⚠  Train or load a model first.")
            else:
                try:
                    predictor.interactive_prediction()
                except Exception as e:
                    print(f"\n❌ Error: {e}")

        elif choice == '4':
            if not predictor.is_trained:
                print("\n⚠  Train or load a model first.")
            else:
                viz.show_feature_importance()

        elif choice == '5':
            viz.show_model_summary()

        elif choice == '6':
            print("\nGoodbye!")
            break

        else:
            print("\n❌ Invalid choice — please enter 1–6.")


if __name__ == "__main__":
    main()


