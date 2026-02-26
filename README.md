<img width="940" height="253" alt="Screenshot 2026-02-26 at 17 40 30" src="https://github.com/user-attachments/assets/d1630f8b-bb6f-4334-9c13-fa8b722c6fb5" />


## Table of Contents
- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Team Members](#team-members)
- [Acknowledgments](#acknowledgments)

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
python main.py
```
You will see a menu:
```
MENU
  1 Train Model
  2 Load Model
  3 Predict Patient
  4 Show patient report 
  5 Exit
```
### Basic Workflow

Select 1 to train a new model (downloads data automatically).

Select 3 to enter patient data and get a prediction (in question other that "age" input 1 for yes and 0 for no).

Select 4 to view visualizations of the prediction.

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
