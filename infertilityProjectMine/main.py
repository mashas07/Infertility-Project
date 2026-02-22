import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('Female infertility.csv')


def plot_patient_distribution(data_column, patient_value, column_name):
    """
        Creates a histogram with a density curve to show the population distribution
        of a specific feature and marks the target patient's position.

        Parameters:
        data_column (Series): The column of data from the fertility dataset (e.g., df['Age']).
        patient_value (float): The specific value belonging to the patient being analyzed.
        column_name (str): The name of the feature for labeling the chart.

        Returns:
        Figure: A matplotlib figure showing the patient's percentile rank in the population.
        """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data_column, bins=20, alpha=0.5, color='mediumseagreen', edgecolor='black', density=True, label='Population')

    kde = stats.gaussian_kde(data_column)
    x_range = np.linspace(data_column.min(), data_column.max(), 100)
    ax.plot(x_range, kde(x_range), 'b-', linewidth=2, label='Population Density')

    ax.axvline(patient_value, color='red', linestyle='--', linewidth=2, label=f'Patient ({column_name}: {patient_value})')

    percentile = stats.percentileofscore(data_column, patient_value)
    ax.text(patient_value, ax.get_ylim()[1] * 0.9, f' {percentile:.1f}th percentile', ha='center', fontsize=12, color='red', weight='bold')
    
    ax.set_xlabel(column_name, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Patient Position in {column_name} Distribution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    return fig

plt.show()


dataset = load_dataset("mstz/fertility", "fertility")["train"]
df = pd.DataFrame(dataset)

label_encoders = {}
for col in df.select_dtypes(include=['string', 'object', 'bool']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df.drop('has_fertility_issues', axis=1)
y = df['has_fertility_issues']
feature_names = X.columns.tolist()

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

patient_data = X.iloc[0].to_dict()


def plot_fertility_analysis(model, feature_names, patient_data):
    """
        Generates a dual-pane visualization comparing global predictive factors
        against a specific patient's health profile.

        Left Plot: Shows the 'Top 10' most influential features determined by
        the Random Forest model (e.g., which habits matter most for fertility).

        Right Plot: Shows the patient's actual recorded values for those same
        top 10 features to see which specific habits might be driving their risk score.

        Args:
            model (RandomForestClassifier): The trained machine learning model.
            feature_names (list): Labels for all analyzed columns.
            patient_data (dict): The raw, non-scaled feature values for the individual.

        Returns:
            matplotlib.figure.Figure: Side-by-side bar charts for medical habit comparison.
        """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.barh(range(len(indices)), importances[indices], color='skyblue', edgecolor='black')
    ax1.set_yticks(range(len(indices)))
    ax1.set_yticklabels(top_features)
    ax1.set_xlabel('Feature Importance', fontsize=12)
    ax1.set_title('Top 10 Most Important Features', fontsize=14, weight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    patient_values = [patient_data.get(f, 0) for f in top_features]
    x_pos = np.arange(len(top_features))

    ax2.barh(x_pos, patient_values, color='coral', alpha=0.7, label='Patient', edgecolor='black')
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(top_features)
    ax2.set_xlabel('Feature Value (Encoded)', fontsize=12)
    ax2.set_title("Patient's Key Feature Values", fontsize=14, weight='bold')
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)

    return fig


fig = plot_fertility_analysis(model, feature_names, patient_data)
plt.tight_layout()
plt.show()


dataset = load_dataset("mstz/fertility", "fertility")["train"]
df = pd.DataFrame(dataset)

le_dict = {}
for col in df.select_dtypes(include=['string', 'object', 'bool']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X = df.drop('has_fertility_issues', axis=1)
y = df['has_fertility_issues']
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

patient_index = 0
patient_scaled = X_scaled[patient_index].reshape(1, -1)
patient_raw = X.iloc[patient_index].to_dict()


def plot_importance_and_values(model, feature_names, patient_data):
    """
        Creates a dual-pane visualization to explain the 'Why' behind a patient's risk score.

        The left plot displays 'Global Importance,' showing which features the AI
        found most influential across the whole dataset. The right plot displays
        'Patient Specific Levels,' showing the actual values for the target patient
        to see how they align with those high-impact factors.

        Args:
            model: The trained RandomForestClassifier.
            feature_names (list): The list of medical/lifestyle factor names.
            patient_data (dict): The specific data row for the patient being analyzed.

        Returns:
            matplotlib.figure.Figure: A side-by-side bar chart comparison.
        """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in indices]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.barh(range(len(indices)), importances[indices], color='skyblue', edgecolor='black')
    ax1.set_yticks(range(len(indices)))
    ax1.set_yticklabels(top_features)
    ax1.set_title('Top 10 Most Important Fertility Factors', weight='bold')
    ax1.invert_yaxis()

    patient_values = [patient_data.get(f, 0) for f in top_features]
    ax2.barh(range(len(top_features)), patient_values, color='coral', alpha=0.7, label='Your Patient')
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features)
    ax2.set_title("Patient's Specific Feature Levels", weight='bold')
    ax2.invert_yaxis()
    ax2.legend()
    plt.tight_layout()
    return fig


def plot_similar_patients(X_train, y_train, patient_scaled):
    """
        Creates a 2D similarity map using PCA to visualize a patient's risk
        profile relative to the rest of the study population.

        This function squashes 10+ medical and lifestyle features into two
        Principal Components (PC1 and PC2). By marking the 20 nearest neighbors,
        it allows for visual 'cluster analysis'—seeing if a patient's data
        points (the Red Star) land among healthy individuals or those with
        fertility issues.

        Args:
            X_train (ndarray): Standardized population feature data.
            y_train (Series): Binary fertility outcomes (1=Yes, 0=No).
            patient_scaled (ndarray): The target patient's data, standardized
                to the same scale as the population.

        Returns:
            matplotlib.figure.Figure: A scatter plot showing population density,
                similarity clusters, and the target patient marker.
        """

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_train)
    patient_2d = pca.transform(patient_scaled)

    nn = NearestNeighbors(n_neighbors=20)
    nn.fit(X_train)
    _, indices = nn.kneighbors(patient_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_train, cmap='viridis', alpha=0.4, label='Population')

    ax.scatter(X_2d[indices[0], 0], X_2d[indices[0], 1], c='orange', s=100, label='20 Most Similar Patients',
               edgecolor='black')

    ax.scatter(patient_2d[0, 0], patient_2d[0, 1], c='red', marker='*', s=500, label='Target Patient', zorder=5)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Fertility Issue (1=Yes, 0=No)')
    ax.set_title('Patient Position Among Similar Cases (PCA)', weight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax.legend()
    return fig


plot_importance_and_values(model, feature_names, patient_raw)
plot_similar_patients(X_scaled, y, patient_scaled)
plt.show()