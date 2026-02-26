<img width="940" height="253" alt="Screenshot 2026-02-26 at 17 40 30" src="https://github.com/user-attachments/assets/d1630f8b-bb6f-4334-9c13-fa8b722c6fb5" />


## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data Interpretation](#data-interpretation)
- [Project Structure](#project-structure)
- [Team Members](#team-members)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Description
This project builds and deploys a Random Forest model to predict female infertility based on patient clinical features. It provides an interactive command-line interface for making predictions, visualizing results, and comparing patient data against dataset averages. The tool is intended as a decision-support aid — not a medical diagnosis.

Built as a course project.

### Features

- **Data Loading & Cleaning**: Automatically downloads the dataset from Kaggle, handles missing values and drops irrelevant columns
- **Model Training**: Trains a Random Forest classifier with 80/20 train/test split and standard scaling
- **Interactive Prediction**: Step-by-step CLI to enter patient data and receive an infertility prediction with confidence score
- **Visualizations**:
  - Prediction probability breakdown by class
  - Patient values vs dataset average comparison
  - Feature contribution chart showing which factors influenced the prediction
- **Feature Importance**: View the top clinical features that drive the model's decisions
- **Model Persistence**: Save and reload trained models to avoid retraining every session


## Installation
### Prerequisites
- Python 3.13 or higher
- All dependencies listed in [`requirements.txt`](requirements.txt)

1. Clone the repository:

```
git clone https://github.com/mashas07/Infertility-Project.git
cd Infertility-Project 
```

2. Install dependencies:

```
pip install -r requirements.txt
```
## Usage
Run the application:
```
python main_file.py
```
You will see a menu:
```
************************************************************
MENU
************************************************************
1. Train new model
2. Load existing model
3. Make prediction
4. Show Patient Report
5. Show model summary
6. Exit

```
### Basic Workflow

### **1. Select 1 to train a new model (downloads data automatically).**
```
Model training
**************************************************
Loaded: 705 rows × 13 columns
Features : 11
Samples  : 705
Train: (564, 11)  |  Test: (141, 11)
Training Random Forest…
MODEL EVALUATION
**************************************************
Accuracy: 93.62%

              precision    recall  f1-score   support

           0       0.94      0.68      0.79        25
           1       0.93      0.99      0.96       116

    accuracy                           0.94       141
   macro avg       0.94      0.84      0.88       141
weighted avg       0.94      0.94      0.93       141

Confusion Matrix:
[[ 17   8]
 [  1 115]]
**************************************************
Saved model → fertility_model.pkl
Saved components → fertility_components.pkl

```

#### Data interpretation
- Features: 11 — the number of clinical variables (columns) used as input to the model, after dropping Patient ID and the target column.
- Samples: 705 — total number of patients in the dataset.
- Train: (564, 11) — 564 patients used to train the model, each with 11 features. This is 80% of 705.
- Test: (141, 11) — 141 patients used to evaluate the model, each with 11 features. This is 20% of 705.
- Model: Random Forest
- Accuracy: 93.62% — correctly classifies 132 out of 141 test patients\

<ins>Class 0 (Fertile) — 25 patients in test set:<ins>

- Precision 0.94 — nearly always right when predicting fertile
- Recall 0.68 — misses 32% of fertile patients, predicting them as infertile
- F1 (score is the balance between precision and recall) 0.79 — moderate, dragged down by low recall
- support = 25 → there are 25 actually fertile patients in the test set

<ins>Class 1 (Infertile) — 116 patients in test set:<ins>

- Precision 0.93 — nearly always right when predicting infertile
- Recall 0.99 — catches 99% of infertile patients
- F1 (score is the balance between precision and recall) 0.96 — excellent overall performance
- support = 116 → there are 116 actually infertile patients in the test set

<ins>Confusion Matrix:<ins>

- 17 correctly fertile, 8 fertile patients misclassified as infertile
- 1 infertile patient missed, 115 correctly infertile


**Macro avg** — calculates the average of precision, recall, and F1 across both classes, treating them equally regardless of how many patients are in each class

**Weighted avg** — calculates the average weighted by how many patients are in each class (25 fertile, 116 infertile)

### **2. Select 2 to use the existing model.**
```
Loaded: 705 rows × 13 columns
Loaded model ← fertility_model.pkl
Loaded components ← fertility_components.pkl
Features: 11
```
When you choose to load model, the program loads two files from your current directory:

- fertility_model.pkl — the saved Random Forest model
- fertility_components.pkl — the saved scaler, feature names, and other components

Which means you can only use this option if you trained the model beforehand.

### **2. Select 3 to enter patient data and get a prediction (in question other that "age" input 1 for yes and 0 for no).**
```
Processing patient data

**************************************************
PREDICTION RESULTS
**************************************************
Prediction : 1
Confidence : 99.57%

  Probability breakdown:
    0                     0.4%
    1                    ================================================= 99.6%

**************************************************
!!  This is a prediction, not a medical diagnosis  !!
Please consult a qualified healthcare professional.
**************************************************

```
- Prediction: 1 — the model predicts this patient is infertile (1 = infertile, 0 = fertile).
- Confidence: 99.57% — the model is extremely confident in this prediction.
Probability breakdown:

- Class 0 (Fertile): 0.4% 
- Class 1 (Infertile): 99.6% 

### **3. Select 4 to view visualizations of the prediction.**

You will see three plots:
- Prediction probability breakdown by class
- Patient values vs dataset average comparison
- Feature contribution chart showing which factors influenced the prediction

You can see an example of a "patient vs. average" plot below:

<img width="1600" height="999" alt="image" src="https://github.com/user-attachments/assets/2c7a39c7-b3d6-4e91-a23c-34a4970638d6" />

### **4. Select 5 to see the model summary**
```
PatientPredictor
  Accuracy : 93.62%
  Features : 11
  Last prediction : 1 (99.6%)
```

## Project Structure
```
Infertility-Project/
├── src/                      # Source code
│   ├── __init__.py
│   ├── DataLoad.py           # Data loading and cleaning
│   ├── Fertility_model.py    # Random Forest model
│   ├── Fertility_predictor.py # Patient prediction interface
│   └── Visualizations.py     # Patient-facing plots
├── main.py                   # Entry point / CLI menu
├── requirements.txt          # Dependencies
├── .gitignore
└── README.md
```

## Team Members

**Masha Sobko (@[mashas07](https://github.com/mashas07))**
- Data loading & preprocessing (DataLoad.py)
- Data cleaning & validation

**Eliška Zsilleova (@[eliskazsilleova](https://github.com/eliskazsilleova))**
- CLI menu & user interaction (main.py)

**Polina Pantechovskaja (@[appollie](https://github.com/appollie))**
- Patient prediction interface (Fertility_predictor.py)

**Zoja Malova (@[toshnotikmustdie](https://github.com/toshnotikmustdie))**
- Model architecture & training (Fertility_model.py)
- Model evaluation & tuning
- Model persistence (save/load)

**Rafaella Tokatlidou (@[RafaellaTk](https://github.com/RafaellaTk))**
- The plots (Visualizations.py)

## Acknowledgments

Dataset: [Female Infertility](https://www.kaggle.com/datasets/fida5073/female-infertility?select=Female+infertility.csv)

Special thanks to our mentor Ruth Großeholz.

## Contact
To report bugs and provide feedback contact the lead developer via m.sobko@student.maastrichtuniversity.nl


Last updated: February 2026
