# ============================================================================
# RANDOM FOREST REGRESSION FOR FERTILITY PREDICTION
# ============================================================================

# 1. IMPORTS AND SETUP
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

class DataProcessor:
    # 2. DATA LOADING AND EXPLORATION
    # ============================================================================
    def load_and_explore_data(filepath):
        """
        Load the fertility dataset and perform initial exploration
        """
        # Load data
        df = pd.read_csv(filepath)

        # Display basic info
        # - df.info()
        # - df.describe()
        # - df.head()
        # - Check for missing values
        # - Check data types

        return df


    # 3. DATA PREPROCESSING
    # ============================================================================
    def preprocess_data(df):
        """
        Clean and prepare data for modeling
        """
        # Handle missing values
        # - Identify columns with missing data
        # - Impute or remove as appropriate

        # Encode categorical variables
        # - Use LabelEncoder or One-Hot Encoding for categorical features
        # - Common columns might include: diagnosis categories, lifestyle factors

        # Feature engineering (if needed)
        # - Create interaction terms
        # - Derive new features from existing ones

        # Separate features (X) and target (y)
        # - Target variable: likely a fertility score/index or outcome measure
        # - Features: all patient characteristics

        return X, y, feature_names

class FertilityModel:
    # 4. TRAIN-TEST SPLIT
    # ============================================================================
    def split_data(X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Optional: Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler


    # 5. MODEL TRAINING
    # ============================================================================
    def train_random_forest(X_train, y_train):
        """
        Train Random Forest Regression model
        """
        # Initialize model with hyperparameters
        rf_model = RandomForestRegressor(
            n_estimators=100,  # Number of trees
            max_depth=None,  # Maximum depth of trees
            min_samples_split=2,  # Minimum samples to split a node
            min_samples_leaf=1,  # Minimum samples in leaf node
            max_features='sqrt',  # Number of features for best split
            random_state=42,
            n_jobs=-1  # Use all processors
        )

        # Fit model
        rf_model.fit(X_train, y_train)

        # Cross-validation for robustness
        cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5,
                                    scoring='neg_mean_squared_error')

        return rf_model


    # 6. MODEL EVALUATION
    # ============================================================================
    def evaluate_model(model, X_test, y_test):
        """
        Evaluate model performance on test set
        """
        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print results
        print(f"Model Performance:")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")

        return y_pred

class PatientPrediction:
    # 7. PATIENT PREDICTION FUNCTION
    # ============================================================================
    def predict_patient_fertility(model, scaler, patient_data, feature_names):
        """
        Predict fertility for a new patient

        Parameters:
        - patient_data: dict with patient characteristics
        - Returns: predicted fertility score
        """
        # Convert patient data to DataFrame
        patient_df = pd.DataFrame([patient_data])

        # Ensure all features are present
        for feature in feature_names:
            if feature not in patient_df.columns:
                patient_df[feature] = 0  # or appropriate default

        # Order columns correctly
        patient_df = patient_df[feature_names]

        # Scale features
        patient_scaled = scaler.transform(patient_df)

        # Predict
        prediction = model.predict(patient_scaled)[0]

        return prediction, patient_scaled

class Visualization:
    # 8. VISUALIZATION FUNCTIONS
    # ============================================================================

    def plot_1_patient_vs_population_distribution(y_train, y_test, patient_prediction):
        """
        PLOT 1: Show where patient falls in population distribution
        Histogram/KDE of all fertility scores with patient's position marked
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Combine train and test for full population
        all_scores = np.concatenate([y_train, y_test])

        # Plot population distribution
        ax.hist(all_scores, bins=30, alpha=0.6, color='skyblue',
                edgecolor='black', density=True, label='Population')

        # Add KDE
        from scipy import stats
        kde = stats.gaussian_kde(all_scores)
        x_range = np.linspace(all_scores.min(), all_scores.max(), 100)
        ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='Population Density')

        # Mark patient's position
        ax.axvline(patient_prediction, color='red', linestyle='--',
                   linewidth=2, label=f'Patient (Score: {patient_prediction:.2f})')

        # Calculate percentile
        percentile = stats.percentileofscore(all_scores, patient_prediction)
        ax.text(patient_prediction, ax.get_ylim()[1] * 0.9,
                f'{percentile:.1f}th percentile',
                ha='center', fontsize=12, color='red', weight='bold')

        ax.set_xlabel('Fertility Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Patient Position in Population Distribution', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        return fig


    def plot_2_feature_importance_with_patient(model, feature_names, patient_data):
        """
        PLOT 2: Feature importance with patient's values overlaid
        Shows which factors are most important and how patient compares
        """
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 features

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # LEFT: Feature importance bar chart
        ax1.barh(range(len(indices)), importances[indices], color='skyblue', edgecolor='black')
        ax1.set_yticks(range(len(indices)))
        ax1.set_yticklabels([feature_names[i] for i in indices])
        ax1.set_xlabel('Feature Importance', fontsize=12)
        ax1.set_title('Top 10 Most Important Features', fontsize=14, weight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # RIGHT: Patient's values for top features compared to population mean
        top_features = [feature_names[i] for i in indices]
        patient_values = [patient_data.get(f, 0) for f in top_features]

        # Would need population means - calculate from training data
        # population_means = X_train_df[top_features].mean()

        x_pos = np.arange(len(top_features))
        ax2.barh(x_pos, patient_values, color='coral', alpha=0.7,
                 label='Patient', edgecolor='black')
        # ax2.scatter(population_means, x_pos, color='blue', s=100,
        #            label='Population Mean', zorder=3)

        ax2.set_yticks(x_pos)
        ax2.set_yticklabels(top_features)
        ax2.set_xlabel('Feature Value', fontsize=12)
        ax2.set_title("Patient's Key Feature Values", fontsize=14, weight='bold')
        ax2.invert_yaxis()
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)

        return fig


    def plot_3_similar_patients_scatter(X_train, y_train, patient_scaled,
                                        patient_prediction, feature_names):
        """
        PLOT 3: 2D scatter showing similar patients
        Use PCA or t-SNE to reduce to 2D, show patient among similar cases
        """
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors

        fig, ax = plt.subplots(figsize=(10, 8))

        # Reduce to 2D using PCA
        pca = PCA(n_components=2, random_state=42)
        X_train_2d = pca.fit_transform(X_train)
        patient_2d = pca.transform(patient_scaled)

        # Find k nearest neighbors
        k = 20
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X_train)
        distances, indices = nn.kneighbors(patient_scaled)

        # Plot all population points
        scatter = ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                             c=y_train, cmap='viridis', alpha=0.5,
                             s=50, edgecolors='black', linewidth=0.5)

        # Highlight similar patients
        ax.scatter(X_train_2d[indices[0], 0], X_train_2d[indices[0], 1],
                   c='orange', s=100, edgecolors='black', linewidth=2,
                   alpha=0.8, label=f'{k} Most Similar Patients')

        # Mark the patient
        ax.scatter(patient_2d[0, 0], patient_2d[0, 1],
                   c='red', marker='*', s=500, edgecolors='black',
                   linewidth=2, label='Your Patient', zorder=5)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Fertility Score', fontsize=12)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        ax.set_title('Patient Position Among Similar Cases', fontsize=14, weight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(alpha=0.3)

        return fig


# 9. MAIN EXECUTION PIPELINE
# ============================================================================
def main():
    """
    Main execution function
    """
    # Load data
    df = load_and_explore_data('Female infertility.csv')

    # Preprocess
    X, y, feature_names = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test, scaler = split_data(X, y)

    # Train model
    model = train_random_forest(X_train, y_train)

    # Evaluate
    y_pred = evaluate_model(model, X_test, y_test)

    # Example: New patient data
    new_patient = {
        'Age': 32,
        'BMI': 24.5,
        # ... other features
    }

    # Predict for new patient
    patient_prediction, patient_scaled = predict_patient_fertility(
        model, scaler, new_patient, feature_names
    )

    print(f"\nPatient's predicted fertility score: {patient_prediction:.2f}")

    # Generate visualizations
    fig1 = plot_1_patient_vs_population_distribution(y_train, y_test, patient_prediction)
    fig2 = plot_2_feature_importance_with_patient(model, feature_names, new_patient)
    fig3 = plot_3_similar_patients_scatter(X_train, y_train, patient_scaled,
                                           patient_prediction, feature_names)

    plt.show()

    return model, scaler, feature_names


if __name__ == "__main__":
    model, scaler, feature_names = main()