"""
Fertility Prediction System
A machine learning system to predict female fertility based on patient data.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
import kagglehub
import matplotlib.pyplot as plt

# Download latest version

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Handles data loading, cleaning, and preprocessing for fertility prediction.
    """

    def __init__(self, csv_path=None):
        """
        Initialize the DataProcessor.

        Args:
            csv_path (str): Path to the CSV file containing fertility data
        """
        self.csv_path = csv_path
        self.data = None
        self.feature_columns = None
        self.target_column = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def load_data(self, csv_path=None):
        """
        Load data from CSV file.

        Args:
            csv_path (str): Path to CSV file. If None, downloads from Kaggle.

        Returns:
            pd.DataFrame: Loaded data
        """
        if csv_path:
            # If a path is provided, use it directly
            self.csv_path = csv_path
        else:
            # Download from Kaggle
            print("Downloading dataset from Kaggle...")
            path = kagglehub.dataset_download("fida5073/female-infertility")
            print(f"Dataset downloaded to: {path}")

            # The path is a directory, so we need to find the CSV file inside it
            if os.path.isdir(path):
                # Find all CSV files in the directory
                csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]

                if not csv_files:
                    raise ValueError(f"No CSV file found in downloaded dataset at {path}")

                # Use the first CSV file found
                self.csv_path = os.path.join(path, csv_files[0])
                print(f"Found CSV file: {csv_files[0]}")
            else:
                # If it's already a file, use it directly
                self.csv_path = path

        if not self.csv_path:
            raise ValueError("CSV path must be provided")

        # Check if the path exists and is a file
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"File not found: {self.csv_path}")

        if os.path.isdir(self.csv_path):
            raise IsADirectoryError(f"Path is a directory, not a file: {self.csv_path}")

        print(f"Loading data from {self.csv_path}...")
        self.data = pd.read_csv(self.csv_path)
        print(f"Data loaded successfully. Shape: {self.data.shape}")

        # Display column names to help identify the target column
        print(f"\nColumns in dataset: {list(self.data.columns)}")

        return self.data
    '''
    # Alternative: If you already know the directory path and just need to fix it
    def load_data_from_directory(directory_path):
        """
        Helper function to load CSV from a known directory

        Args:
            directory_path (str): Path to directory containing CSV

        Returns:
            pd.DataFrame: Loaded data
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        # Find CSV files
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

        if not csv_files:
            raise ValueError(f"No CSV files found in {directory_path}")

        print(f"Found {len(csv_files)} CSV file(s):")
        for i, f in enumerate(csv_files, 1):
            print(f"  {i}. {f}")

        # Use the first CSV file
        csv_path = os.path.join(directory_path, csv_files[0])
        print(f"\nUsing: {csv_files[0]}")

        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        return df
    '''
    def clean_data(self, drop_threshold=0.3):
        """
        Clean the dataset by handling missing values and outliers.

        Args:
            drop_threshold (float): Threshold for dropping columns with too many missing values
az
        Returns:
            pd.DataFrame: Cleaned data
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")

        print("\nCleaning data...")
        initial_shape = self.data.shape

        # Drop columns with too many missing values
        missing_ratio = self.data.isnull().sum() / len(self.data)
        cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index
        if len(cols_to_drop) > 0:
            print(f"Dropping {len(cols_to_drop)} columns with >{drop_threshold * 100}% missing values")
            self.data = self.data.drop(columns=cols_to_drop)

        # Fill missing values for numerical columns with median
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].median(), inplace=True)

        # Fill missing values for categorical columns with mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().any():
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)

        # Remove duplicate rows
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            print(f"Removing {duplicates} duplicate rows")
            self.data = self.data.drop_duplicates()

        print(f"Data cleaned. Shape changed from {initial_shape} to {self.data.shape}")
        return self.data

    def prepare_features(self, target_col):
        """
        Prepare features and target variable for model training.

        Args:
            target_col (str): Name of the target column

        Returns:
            tuple: (X, y) features and target
        """
        if self.data is None:
            raise ValueError("Data must be loaded first")

        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        print(f"\nPreparing features with target: {target_col}")
        self.target_column = target_col

        # Separate features and target
        X = self.data.drop(columns=[target_col])
        y = self.data[target_col]

        # Encode categorical variables in features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

        # Encode target variable if it's categorical
        if y.dtype == 'object':
            y = self.label_encoder.fit_transform(y)
            print(f"Target classes: {self.label_encoder.classes_}")

        self.feature_columns = X.columns.tolist()
        print(f"Features prepared. Shape: {X.shape}")

        return X, y

    def scale_features(self, X_train, X_test=None):
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Test features (optional)

        Returns:
            Scaled features
        """
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)

        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def get_data_info(self):
        """Display information about the loaded data."""
        if self.data is None:
            print("No data loaded")
            return

        print("\n" + "=" * 60)
        print("DATA INFORMATION")
        print("=" * 60)
        print(f"\nDataset shape: {self.data.shape}")
        print(f"\nColumn names and types:")
        print(self.data.dtypes)
        print(f"\nMissing values:")
        print(self.data.isnull().sum())
        print(f"\nBasic statistics:")
        print(self.data.describe())
        print("=" * 60)


class FertilityModel:
    """
    Machine learning model for fertility prediction.
    """

    def __init__(self, model_type='random_forest'):
        """
        Initialize the fertility prediction model.

        Args:
            model_type (str): Type of model to use
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_names = None

        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Model type '{model_type}' not supported")

    def train(self, X_train, y_train):
        """
        Train the model on training data.

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\nTraining {self.model_type} model...")
        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Model training completed!")

    def predict(self, X):
        """
        Make predictions on new data.

        Args:
            X: Features to predict

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Get prediction probabilities.

        Args:
            X: Features to predict

        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nAccuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("=" * 60)

        return {
            'accuracy': accuracy,
            'predictions': y_pred
        }

    def get_feature_importance(self):
        """
        Get feature importance scores.

        Returns:
            pd.DataFrame: Feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if not hasattr(self.model, 'feature_importances_'):
            print("This model doesn't support feature importance")
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'feature_{i}' for i in
                                                                      range(len(self.model.feature_importances_))],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return importance_df

    def save_model(self, filepath):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        self.is_trained = True
        print(f"\nModel loaded from {filepath}")


class FertilityPredictor:
    def __init__(self):
        self.encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def run_pipeline(self, csv_path, target_column):
        # 1. Load
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # Remove hidden spaces

        print(f"Data Loaded. Shape: {df.shape}")

        # 2. Validate Target
        if target_column not in df.columns:
            print(f"\n❌ Error: Column '{target_column}' not found.")
            print(f"Available columns are: {list(df.columns)}")
            return

        # 3. Clean
        # Fill numeric missing values with median
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())

        # 4. Prepare Features
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Convert categorical features to numeric
        X = pd.get_dummies(X, drop_first=True)

        # Encode target if it's text
        if y.dtype == 'object':
            y = self.encoder.fit_transform(y)

        # 5. Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 6. Scale & Train
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Training Random Forest...")
        self.model.fit(X_train_scaled, y_train)

        # 7. Evaluate
        predictions = self.model.predict(X_test_scaled)
        print("\n" + "=" * 30)
        print(f"RESULTS FOR: {target_column}")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
        print("=" * 30)
        print(classification_report(y_test, predictions))

    import scipy.stats as stats
    def get_user_input_and_plot(self):
        print("\n--- Fertility Assessment Input ---")
        user_data = {}

        # We'll ask for the most common/important features found in the dataset
        # Note: In a real app, you'd match every column in self.feature_names
        try:
            user_data['Age'] = float(input("Enter your Age: "))
            user_data['BMI'] = float(input("Enter your BMI: "))
            user_data['Cycle length(days)'] = float(input("Enter average Cycle Length (days): "))
            # Add placeholders for other features to match model shape
            for col in self.feature_names:
                if col not in user_data:
                    user_data[col] = 0  # Defaulting unknown values to 0

            # Convert to DataFrame and Scale
            user_df = pd.DataFrame([user_data])[self.feature_names]
            user_scaled = self.scaler.transform(user_df)

            # Calculate Risk (Probability of Disorder)
            risk_score = self.model.predict_proba(user_scaled)[0, 1]
            fertility_score = (1 - risk_score) * 100

            self.plot_distribution(risk_score, fertility_score)

        except ValueError:
            print("Please enter valid numerical values.")

    def plot_distribution(self, user_risk, fertility_score):
        plt.figure(figsize=(10, 6))

        # Plot the background distribution of the population
        sns.kdeplot(self.population_probs, fill=True, color="skyblue", label="Population Data")

        # Mark the user
        plt.axvline(user_risk, color='red', linestyle='--', linewidth=2)
        plt.text(user_risk + 0.02, 0.5, f'YOU: {fertility_score:.1f}% Fertility Score', color='red', fontweight='bold')

        plt.title('Where you stand: Fertility Risk Distribution')
        plt.xlabel('Risk Level (Probability of Ovulation Disorder)')
        plt.ylabel('Patient Density')
        plt.legend()
        plt.show()

        percentile = percentileofscore(self.population_probs, user_risk)
        print(f"\nYour Fertility Score: {fertility_score:.1f}/100")
        print(f"You have a lower risk than {100 - percentile:.1f}% of the population in this study.")



def main():
    # Attempt to find the Kaggle path automatically
    expected_path = '/Users/mashasobko/.cache/kagglehub/datasets/fida5073/female-infertility/versions/1'

    if not os.path.exists(expected_path):
        print("Dataset not found locally. Downloading...")
        expected_path = kagglehub.dataset_download("fida5073/female-infertility")

    # Find the first CSV in that folder
    files = [f for f in os.listdir(expected_path) if f.endswith('.csv')]
    if not files:
        print("No CSV files found in the directory.")
        return

    full_path = os.path.join(expected_path, files[0])

    # Initialize and run
    predictor = FertilityPredictor()

    # NOTE: Common targets in this dataset are 'PCOS (Y/N)' or 'Infertility'
    # Change 'PCOS (Y/N)' below if your specific CSV uses a different header
    predictor.run_pipeline(full_path, target_column='Ovulation Disorders')

if __name__ == "__main__":
    main()
