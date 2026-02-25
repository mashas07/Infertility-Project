import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.DataLoad import DataLoader

class FertilityModel:
    '''Creates and trains a model'''
    def __init__(self):
        self._loader = DataLoader()
        self._model: RandomForestClassifier | None = None
        self._scaler = StandardScaler()
        self._feature_names: list[str] = []
        self._original_features: list[str] = []
        self._target_classes = None
        self._accuracy: float | None = None
        self._is_trained: bool = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def accuracy(self) -> float:
        if self._accuracy is None:
            raise RuntimeError("Accuracy cannot be None")
        else:
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

    def train(self, target_column: str = 'Infertility Prediction'):
        '''Load and clean the data.
            Train and evaluate the model.'''
        print("Model training")
        print(f"{"*" * 50}")

        self._loader.load()
        self.loader.clean()
        X, y = self._loader.split_features_target(target_column)
        self._original_features = X.columns.tolist()
        self._feature_names = self._original_features.copy()

        print(f"Features : {len(self._original_features)}")
        print(f"Samples  : {len(X)}")

        # All values in this dataset are numeric, so encoding isn't needed
        X_values = X.values
        y_values = y.values

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_values, y_values, test_size=0.2, random_state=42, stratify=y_values
        )
        print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

        # Scale
        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s = self._scaler.transform(X_test)

        print("Training Random Forest…")
        self._model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10, min_samples_split=5
        )
        self._model.fit(X_train_s, y_train)
        self._is_trained = True

        # Evaluate
        y_pred = self._model.predict(X_test_s)
        self._accuracy = accuracy_score(y_test, y_pred)

        # Store target classes if needed for decoding
        self._target_classes = self._model.classes_

        print("MODEL EVALUATION")
        print(f"{'*' * 50}")
        print(f"Accuracy: {self._accuracy * 100:.2f}%\n")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print(f"{'*' * 50}")

        return self._accuracy

    def predict_encoded(self, user_df: pd.DataFrame):
        X_scaled = self._scaler.transform(user_df[self._original_features].values)
        prediction = self._model.predict(X_scaled)[0]
        probabilities = self._model.predict_proba(X_scaled)[0]
        return prediction, probabilities

    def decode_prediction(self, prediction, probabilities):
        predicted_class = prediction
        confidence = float(max(probabilities)) * 100
        prob_dict = {str(cls): probabilities[i] * 100
                     for i, cls in enumerate(self._target_classes)}

        return predicted_class, confidence, prob_dict

    def feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        return (pd.DataFrame(
            {"feature": self._feature_names,
            "importance": self._model.feature_importances_,
        }).sort_values(by="importance", ascending=False).head(top_n).reset_index(drop=True)
        )

    def save(self, model_path: str = 'fertility_model.pkl',
             components_path: str = 'fertility_components.pkl'):
        '''Persist model and components to disk.'''
        if not self._is_trained:
            raise RuntimeError("Train the model before saving.")
        with open(model_path, 'wb') as f:
            pickle.dump(self._model, f)
        components = {
            'scaler': self._scaler,
            'feature_names': self._feature_names,
            'original_features': self._original_features,
            'target_classes': self._target_classes,
            'accuracy': self._accuracy,
        }
        with open(components_path, 'wb') as f:
            pickle.dump(components, f)
        print(f"Saved model → {model_path}")
        print(f"Saved components → {components_path}")

    def load(self, model_path: str = 'fertility_model.pkl',
             components_path: str = 'fertility_components.pkl'):
        '''Restore model and components from disk.'''
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
        with open(components_path, 'rb') as f:
            c = pickle.load(f)
        self._scaler = c['scaler']
        self._feature_names = c['feature_names']
        self._original_features = c['original_features']
        self._target_classes = c['target_classes']
        self._accuracy = c['accuracy']
        self._is_trained = True
        self._loader.load()
        self._loader.clean()
        print(f"Loaded model ← {model_path}")
        print(f"Loaded components ← {components_path}")
        print(f"Features: {len(self._original_features)}")

