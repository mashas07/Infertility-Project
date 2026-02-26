import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datasets import load_dataset

sns.set_theme(style="whitegrid", palette="muted")

class FertilityModel:
    def __init__(self):
        raw = load_dataset("mstz/fertility", "fertility")["train"]
        self.loader = type('obj', (object,), {'dataframe' : pd.DataFrame(raw)})()
        
        df = self.loader.dataframe
        self.original_features = df.columns.drop('has_fertility_issues').tolist()
        
        self.X_train = df[self.original_features].copy()
        self.y_train = df['has_fertility_issues']

        for col in self.X_train.select_dtypes(include=['object', 'string', 'bool']).columns:
            le = LabelEncoder()
            self.X_train[col] = le.fit_transform(self.X_train[col].astype(str))

        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def feature_importance(self, top_n=8):
        """Returns which features are most important."""
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        return pd.DataFrame({
            'feature': [self.original_features[i] for i in indices],
            'importance': importances[indices]
        })


class Visualization:
    def __init__(self, model: FertilityModel):
        if not isinstance(model, FertilityModel):
            raise TypeError("model must be a FertilityModel instance.")
        self._model = model

    def plot_prediction_probability(self, result: dict):
        """Bar chart of predicted class probabilities."""
        probs = result.get('probabilities')
        if not probs: return

        classes = list(probs.keys())
        values = [probs[c] for c in classes]
        colors = ['#d9534f' if c == str(result['prediction']) else '#5bc0de' for c in classes]

        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(classes, values, color=colors, edgecolor='white', height=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=11)

        ax.set_xlim(0, 115)
        ax.set_xlabel('Probability (%)')
        ax.set_title('Prediction Probability', fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_patient_vs_average(self, result: dict, top_n: int = 8):
        """Grouped bar chart comparing patient values against averages."""
        patient_data = result.get('patient_data')
        df = self._model.loader.dataframe
        features = self._model.original_features[:top_n]

        dataset_means = df[features].mean()
        patient_vals = pd.Series({f: float(patient_data[f]) for f in features})

        combined_max = pd.concat([dataset_means, patient_vals]).groupby(level=0).max().replace(0, 1)
        norm_avg = dataset_means / combined_max
        norm_pat = patient_vals / combined_max

        x = np.arange(len(features))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, norm_avg, width, label='Dataset Average', color='#5bc0de')
        ax.bar(x + width / 2, norm_pat, width, label='Your Values', color='#d9534f')

        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=30, ha='right')
        ax.set_title('Your Values vs Dataset Average', fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_patient_report(self, result: dict):
        """Show all patient plots."""
        self.plot_prediction_probability(result)
        self.plot_patient_vs_average(result)

my_model = FertilityModel()

viz = Visualization(my_model)

patient_idx = 0
raw_patient_row = my_model.loader.dataframe.iloc[patient_idx]

patient_features_only = my_model.X_train.iloc[[patient_idx]]
probs = my_model.model.predict_proba(patient_features_only)[0]

analysis_result = {
    'prediction': 'Issues' if probs[1] > 0.5 else 'Normal',
    'probabilities': {'Normal': probs[0] * 100, 'Issues': probs[1] * 100},
    'patient_data': raw_patient_row.to_dict()
}

viz.plot_patient_report(analysis_result)
