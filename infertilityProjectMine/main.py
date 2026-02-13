import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv('Female infertility.csv')

def plot_patient_distribution(data_column, patient_value, column_name):
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
plot_patient_distribution(df['Age'], 30, 'Age')
plt.show()