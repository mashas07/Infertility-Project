"""
Visualization.py
Patient-facing plots built with matplotlib and seaborn.
Composes FertilityModel (or subclass) to access model internals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from Fertility_model import FertilityModel

sns.set_theme(style="whitegrid", palette="muted")


class Visualization:
    """
    Patient-facing plots.
    Composes a FertilityModel (or subclass) to access model internals.
    """

    def __init__(self, model: FertilityModel):
        if not isinstance(model, FertilityModel):
            raise TypeError("model must be a FertilityModel instance.")
        self._model = model

    # ── magic methods ────────────────────────────

    def __repr__(self):
        return f"Visualization(model={self._model!r})"

    def __str__(self):
        return f"Visualization → bound to {self._model.__class__.__name__}"

    # ── patient plots ────────────────────────────

    def plot_prediction_probability(self, result: dict):
        """Bar chart of predicted class probabilities."""
        probs = result.get('probabilities')
        if not probs:
            print("    No probability data available.")
            return

        classes = list(probs.keys())
        values = [probs[c] for c in classes]
        colors = ['#d9534f' if c == str(result['prediction']) else '#5bc0de' for c in classes]

        fig, ax = plt.subplots(figsize=(8, max(3, len(classes) * 0.8)))
        bars = ax.barh(classes, values, color=colors, edgecolor='white', height=0.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=11)

        ax.set_xlim(0, 115)
        ax.set_xlabel('Probability (%)', fontsize=12)
        ax.set_title('Prediction Probability by Class', fontsize=14, fontweight='bold')

        predicted_patch = mpatches.Patch(color='#d9534f', label=f'Predicted: {result["prediction"]}')
        other_patch = mpatches.Patch(color='#5bc0de', label='Other classes')
        ax.legend(handles=[predicted_patch, other_patch], loc='lower right')

        ax.text(0.5, -0.15,
                '  This is a prediction, not a medical diagnosis. Please consult a doctor.',
                ha='center', transform=ax.transAxes, fontsize=9, color='gray', style='italic')

        plt.tight_layout()
        plt.show()

    def plot_patient_vs_average(self, result: dict, top_n: int = 8):
        """Grouped bar chart comparing patient values against dataset averages."""
        patient_data = result.get('patient_data')
        if not patient_data:
            print("    No patient data in result.")
            return

        try:
            df = self._model.loader.dataframe
        except RuntimeError as e:
            print(f"  Dataset error: {e}")

            return

        features = self._model.original_features[:top_n]
        features = [f for f in features if f in df.columns]

        dataset_means = df[features].mean()
        patient_vals = pd.Series({f: float(patient_data[f]) for f in features})

        # Normalise to 0-1 for fair comparison
        combined_max = pd.concat([dataset_means, patient_vals]).groupby(level=0).max().replace(0, 1)
        norm_avg = dataset_means / combined_max
        norm_pat = patient_vals / combined_max

        x = np.arange(len(features))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, norm_avg, width, label='Dataset Average', color='#5bc0de', edgecolor='white')
        ax.bar(x + width / 2, norm_pat, width, label='Your Values', color='#d9534f', edgecolor='white')

        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=30, ha='right', fontsize=10)
        ax.set_ylabel('Normalised Value', fontsize=12)
        ax.set_title('Your Values vs Dataset Average', fontsize=14, fontweight='bold')
        ax.legend()

        ax.text(0.5, -0.25,
                '  Values are normalised for comparison. Consult a healthcare professional.',
                ha='center', transform=ax.transAxes, fontsize=9, color='gray', style='italic')

        plt.tight_layout()
        plt.show()

    def plot_feature_contribution(self, result: dict, top_n: int = 8):
        """Diverging bar chart showing which features drove the prediction."""
        patient_data = result.get('patient_data')
        if not patient_data:
            print("    No patient data in result.")
            return

        try:
            df = self._model.loader.dataframe
        except RuntimeError:
            print("    Dataset not available for contribution plot.")
            return

        importance_df = self._model.feature_importance(top_n=top_n)
        features = [f for f in importance_df['feature'] if f in df.columns]

        means = df[features].mean()
        stds = df[features].std().replace(0, 1)
        importances = importance_df.set_index('feature')['importance']

        contributions = {
            f: ((float(patient_data[f]) - means[f]) / stds[f]) * importances.get(f, 0)
            for f in features
        }

        contrib_series = pd.Series(contributions).sort_values()
        colors = ['#d9534f' if v > 0 else '#5cb85c' for v in contrib_series]

        fig, ax = plt.subplots(figsize=(9, max(4, len(contrib_series) * 0.6)))
        contrib_series.plot(kind='barh', ax=ax, color=colors, edgecolor='white')

        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel('Contribution to Prediction\n(positive = increases risk, negative = decreases risk)',
                      fontsize=11)
        ax.set_title('Which Factors Influenced Your Prediction Most', fontsize=14, fontweight='bold')

        increase = mpatches.Patch(color='#d9534f', label='Increases predicted risk')
        decrease = mpatches.Patch(color='#5cb85c', label='Decreases predicted risk')
        ax.legend(handles=[increase, decrease])

        ax.text(0.5, -0.18,
                '  This is an approximation. Please consult a qualified healthcare professional.',
                ha='center', transform=ax.transAxes, fontsize=9, color='gray', style='italic')

        plt.tight_layout()
        plt.show()

    def plot_patient_report(self, result: dict):
        """Show all 3 patient plots at once."""
        self.plot_prediction_probability(result)
        self.plot_patient_vs_average(result)
        self.plot_feature_contribution(result)