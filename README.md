# XAI Pioneers - Phishing Website Detection

This project implements various machine learning models for phishing website detection, with a focus on explainable AI techniques. The project is structured as a series of Jupyter notebooks that handle different aspects of the machine learning pipeline.

## Project Structure

The project is organized as follows:

1. **Data Preprocessing and Analysis**
   - `Team 30 XAI Pioneers EDA.ipynb`: Exploratory Data Analysis
   - `Team 30 XAI Pioneers Preprocessing.ipynb`: Data preprocessing steps
   - `Team 30 XAI Pioneers Feature Engineering.ipynb`: Feature engineering pipeline

2. **Model Implementation Notebooks**
   - `Team_30_XAI_Pioneers_Logistic_Regression_Youssef_ElDawayaty.ipynb`
   - `Team_30_XAI_Pioneers_DT_Abdulrahman_Hosny.ipynb`
   - `Team_30_XAI_Pioneers_RF_Abdulrahman_Hosny.ipynb`
   - `Team_30_XAI_Pioneers_XGboost_Abdulrahman_Hosny.ipynb`
   - `Team_30_XAI_Pioneers_Naive_Bayes_Mariam_Hani.ipynb`
   - `Team_30_XAI_Pioneers_MLP_Mariam_Hani.ipynb`
   - `Team 30 XAI Pioneers SVM Mariam Hani.ipynb`
   - `Team_30_XAI_Pioneers__GBM_Youssef_ElDawayaty.ipynb`
   - `Team_30_XAI_Pioneers__StackingClassifier_Youssef_ElDawayaty.ipynb`
   - `Team_30_XAI_Pioneers_KNN_AhmedSameh.ipynb`
   - `Team_30_XAI_Pioneers_LSD_(Hard_Voting)_AhmedSameh.ipynb`
   - `Team_30_XAI_Pioneers_LSD_(Soft_Voting)_AhmedSameh.ipynb`

3. **Data Files**
   - `Phishing Websites Preprocessed.csv`: Preprocessed dataset
   - `Phishing Websites Engineered.csv`: Feature-engineered dataset

## Prerequisites

To run this project, you'll need:

1. Python 3.7 or higher
2. Jupyter Notebook or JupyterLab
3. Required Python packages (install using pip):
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm tensorflow keras shap lime
   ```

## How to Run the Code

1. **Setup Environment**
   ```bash
   # Create a virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install required packages
   pip install -r requirements.txt
   ```

2. **Running the Notebooks**
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open the notebooks in the following order:
     1. First run the EDA notebook to understand the data
     2. Then run the preprocessing notebook
     3. Follow with the feature engineering notebook
     4. Finally, run any of the model implementation notebooks

3. **Data Flow**
   - The preprocessing notebook generates `Phishing Websites Preprocessed.csv`
   - The feature engineering notebook uses the preprocessed data to generate `Phishing Websites Engineered.csv`
   - The model notebooks use the engineered dataset for training and evaluation

## Model Implementations

The project includes implementations of various machine learning models:
- Logistic Regression
- Decision Trees
- Random Forest
- XGBoost
- Naive Bayes
- Multi-layer Perceptron (MLP)
- Support Vector Machine (SVM)
- Gradient Boosting Machine (GBM)
- Stacking Classifier
- K-Nearest Neighbors (KNN)
- LSD (Hard and Soft Voting)

Each model notebook includes:
- Model training and evaluation
- Performance metrics
- Explainable AI techniques
- Visualization of results

## Reports

Detailed reports and analysis can be found in the `Reports/` directory.

## Contributing

This project was developed by Team 30 XAI Pioneers:
- Abdulrahman Hosny
- Mariam Hani
- Youssef ElDawayaty
- Ahmed Sameh

## License

This project is licensed under the MIT License - see the LICENSE file for details. 