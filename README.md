# XAI Pioneers - Phishing Website Detection

This project implements various machine learning models for phishing website detection, with a focus on Explainable AI (XAI) techniques. The project includes data preprocessing, feature engineering, and multiple model implementations.

## Project Structure

The project is organized into several Jupyter notebooks, each focusing on a specific aspect of the analysis:

1. **Data Exploration and Preprocessing**
   - `Team 30 XAI Pioneers EDA.ipynb`: Exploratory Data Analysis
   - `Team 30 XAI Pioneers Preprocessing.ipynb`: Data preprocessing steps
   - `Team 30 XAI Pioneers Feature Engineering.ipynb`: Feature engineering pipeline

2. **Model Implementations**
   - `Team 30 XAI Pioneers DT Abdulrahman Hosny.ipynb`: Decision Tree model
   - `Team 30 XAI Pioneers RF Abdulrahman Hosny.ipynb`: Random Forest model
   - `Team 30 XAI Pioneers Naive Bayes Mariam Hani.ipynb`: Naive Bayes model
   - `Team 30 XAI Pioneers SVM Mariam Hani.ipynb`: Support Vector Machine model
   - `Team_30_XAI_Pioneers_Logistic_Regression_Youssef_ElDawayaty.ipynb`: Logistic Regression model
   - `Team_30_XAI_Pioneers__GBM_Youssef_ElDawayaty.ipynb`: Gradient Boosting Machine model
   - `Team  30 XAI Pioneers LSD AhmedSameh.ipynb`: Linear Discriminant Analysis model

3. **Data Files**
   - `Phishing Websites Preprocessed.csv`: Preprocessed dataset
   - `Phishing Websites Engineered.csv`: Feature-engineered dataset

4. **Reports**
   - Contains detailed reports and analysis documentation

## Prerequisites

To run this project, you'll need:

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages (install using `pip install -r requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - shap (for XAI)
  - lime (for XAI)

## Getting Started

1. **Clone the repository**
   ```bash
   git clone [repository-url]
   cd XAI-Pioneers
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks in sequence**
   The notebooks should be run in the following order:
   1. EDA notebook
   2. Preprocessing notebook
   3. Feature Engineering notebook
   4. Any of the model implementation notebooks

## Running the Models

Each model notebook can be run independently after the preprocessing and feature engineering steps are completed. The notebooks contain:
- Model implementation
- Training and evaluation
- XAI techniques for model interpretation
- Performance metrics and visualizations

## Data

The project uses two main datasets:
- `Phishing Websites Preprocessed.csv`: Contains the preprocessed data
- `Phishing Websites Engineered.csv`: Contains the feature-engineered data

Make sure these files are in the root directory of the project before running the notebooks.

## Contributing

This project was developed by Team 30 XAI Pioneers. For any questions or contributions, please contact the team members.

## Team Members

- Abdulrahman Hosny
- Mariam Hani
- Youssef ElDawayaty
- Ahmed Sameh 