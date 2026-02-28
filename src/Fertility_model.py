import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.DataLoad import DataLoader

class FertilityModel:
    '''
    A machine learning model for predicting infertility using a Random Forest classifier.

    Handles data loading, preprocessing, training, evaluation, prediction,
    and persistence (save/load) of the model and its components.
    '''
    def __init__(self):
        '''Initialize the FertilityModel with default (untrained) state.'''
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
        '''Return whether the model has been trained.'''
        return self._is_trained

    @property
    def accuracy(self) -> float:
        '''
        Return the model's test-set accuracy.

        Raises:
            RuntimeError: If the model has not been trained yet.
        '''
        if self._accuracy is None:
            raise RuntimeError("Accuracy cannot be None")
        return self._accuracy

    @property
    def feature_names(self) -> list[str]:
        '''Return the list of feature names used by the model.'''
        return self._feature_names

    @property
    def original_features(self) -> list[str]:
        '''Return the original feature names from the training dataset.'''
        return self._original_features

    @property
    def loader(self) -> DataLoader:
        '''Return the DataLoader instance used for loading and cleaning data.'''
        return self._loader

    def __str__(self):
        '''Return a human-readable summary of the model's state and performance.'''
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
        '''
        Load, clean, and split the data, then train and evaluate the Random Forest model.

        Applies stratified train/test split to handle class imbalance, standardizes
        features, and prints accuracy, a classification report, and a confusion matrix.

        Args:
            target_column: Name of the column to use as the prediction target.

        Returns:
            The accuracy score achieved on the test set.
        '''
        print("Model training")
        print(f"{"*" * 50}")

        self._loader.load()
        self.loader.clean()
        X, y = self._loader.split_features_target(target_column)
        self._original_features = X.columns.tolist()
        self._feature_names = self._original_features.copy()

        print(f"Features : {len(self._original_features)}")
        print(f"Samples  : {len(X)}")

        # Extract matrix of values from the dataframe
        X_values = X.values
        y_values = y.values

        # Use stratified train-test split in case data in y is unbalanced
        X_train, X_test, y_train, y_test = train_test_split(
            X_values, y_values, test_size=0.2, random_state=42, stratify=y_values
        )
        print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

        # Apply standardization to the train data only
        X_train_s = self._scaler.fit_transform(X_train)
        X_test_s = self._scaler.transform(X_test)

        self._model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        self._model.fit(X_train_s, y_train)
        self._is_trained = True

        # Evaluation
        y_pred = self._model.predict(X_test_s)
        self._accuracy = accuracy_score(y_test, y_pred)

        # Store target classes if needed for decoding
        self._target_classes = self._model.classes_

        print("MODEL EVALUATION")
        print(f"{'*' * 50}")
        print(f"Accuracy: {self._accuracy * 100:.2f}%\n")
        # Calculates and prints different metrics for model evaluation(accuracy, precision etc
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        # Rows = actual class, Columns = predicted class
        # Matrix with amounts of true negatives, false negatives, false positives, true positives
        print(confusion_matrix(y_test, y_pred))
        print(f"{'*' * 50}")

        return self._accuracy

    # Method for predicting user input
    def predict_encoded(self, user_df: pd.DataFrame):
        '''
        Scale user input and return the raw model prediction and class probabilities.

        Args:
            user_df: DataFrame containing user-provided feature values.

        Returns:
            A tuple of (predicted_class_label, probability_array).
        '''
        X_scaled = self._scaler.transform(user_df[self._original_features].values)
        prediction = self._model.predict(X_scaled)[0]
        probabilities = self._model.predict_proba(X_scaled)[0]
        return prediction, probabilities

    # Calculate prediction confidence and probabilities for all classes(in our case for 0 or 1)
    def decode_prediction(self, prediction, probabilities):
        '''
        Convert raw model output into a human-readable prediction result.

        Args:
            prediction: The predicted class label returned by the model.
            probabilities: Array of class probabilities from predict_proba.

        Returns:
            A tuple of (predicted_class, confidence_percentage, prob_dict), where
            prob_dict maps each class label (as a string) to its probability percentage.
        '''
        predicted_class = prediction
        confidence = float(max(probabilities)) * 100
        prob_dict = {str(cls): probabilities[i] * 100
                     for i, cls in enumerate(self._target_classes)}

        return predicted_class, confidence, prob_dict

    # Returns how important was each feature for our prediction
    def feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        '''
        Return the top N most important features used by the trained model.

        Args:
            top_n: Number of top features to return (default 10).

        Returns:
            A DataFrame with columns 'feature' and 'importance', sorted descending.

        Raises:
            RuntimeError: If the model has not been trained yet.
        '''
        if not self._is_trained:
            raise RuntimeError("Model must be trained before making predictions")

        return (pd.DataFrame(
            {"feature": self._feature_names,
            "importance": self._model.feature_importances_,
        }).sort_values(by="importance", ascending=False).head(top_n).reset_index(drop=True)
        )

    # Saves model to the disk
    def save(self, model_path: str = 'fertility_model.pkl',
             components_path: str = 'fertility_components.pkl'):
        '''
        Persist the trained model and its preprocessing components to disk.

        Args:
            model_path: File path for saving the Random Forest model.
            components_path: File path for saving the scaler, feature names,
                             target classes, and accuracy.

        Raises:
            RuntimeError: If the model has not been trained yet.
        '''
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
        print(f"Saved model -> {model_path}")
        print(f"Saved components -> {components_path}")

    def load(self, model_path: str = 'fertility_model.pkl',
             components_path: str = 'fertility_components.pkl'):
        '''
        Restore a previously saved model and its components from disk.

        Also reloads and cleans the dataset via the DataLoader so the instance
        is fully ready for predictions without retraining.

        Args:
            model_path: File path of the saved Random Forest model.
            components_path: File path of the saved components (scaler, features, etc.).
        '''
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
        print(f"Loaded model <- {model_path}")
        print(f"Loaded components <- {components_path}")
        print(f"Features: {len(self._original_features)}")

