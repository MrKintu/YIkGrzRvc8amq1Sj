# YIkGrzRvc8amq1Sj
Customer Happiness Prediction Model
This repository contains the code to build a machine-learning model that predicts whether a customer is happy or not based on responses from a survey. The dataset includes various features that represent customer feedback, and the goal is to classify customers into two categories: happy and not happy.

Overview
The project involves the following tasks:

Data Preprocessing: Preparing and cleaning the survey data by handling missing values, class imbalance, and scaling features.
Model Training: Using machine learning models to predict customer happiness based on survey responses.
Hyperparameter Tuning: Optimizing the model to achieve the best possible performance using techniques like GridSearchCV.
Feature Selection: Identifying the most important survey questions/features that impact the prediction, helping to determine which features could be removed to streamline future surveys without sacrificing model performance.
Model Evaluation: Evaluate model performance using accuracy, classification report, ROC-AUC score, and confusion matrix to ensure it performs well on unseen data.
Requirements
The following libraries are required to run the code:

pandas
numpy
scikit-learn
imblearn
matplotlib
seaborn
You can install these dependencies using the following command:

bash
Copy code
pip install -r requirements.txt
Dataset
The dataset used in this project is a survey of customers, where each row represents a customer’s responses to a set of questions. The target variable (Y) indicates whether a customer is happy (1) or not happy (0), and the features (X1, X2, X3, etc.) represent survey question answers.

Steps Involved
1. Data Exploration and Preprocessing
The script begins by loading the dataset and exploring its structure, including missing values, duplicates, and basic statistics. The data is then split into features (X) and the target variable (Y).

2. Data Visualisation
The data is visualized using plots to understand the distribution of responses and feature relationships. This includes:

Bar plots of the target variable.
Box plots for feature distribution.
Pair plots and correlation heatmaps to understand the relationships between features.

3. Model Definition and Training
A Random Forest Classifier is used to build the prediction model. The model is trained on the preprocessed data, with class imbalance handled by oversampling using SMOTE.

4. Hyperparameter Tuning
GridSearchCV is used to perform hyperparameter tuning to find the best model parameters for the Random Forest Classifier.

5. Feature Importance and Selection
The importance of each feature in predicting customer happiness is calculated, and the top features are visualized using a bar plot. Additionally, Recursive Feature Elimination with Cross-Validation (RFECV) is used to identify the minimal set of features that provide the most predictive power.

6. Model Evaluation
The model is evaluated on the test data using various metrics, including:

Accuracy Score
Classification Report
ROC-AUC Score
Confusion Matrix

7. Insights and Conclusions
The script provides insights into the most important features for predicting customer happiness and discusses how a reduced set of features can simplify the model while maintaining performance.

How to Run
Clone the repository to your local machine:
git clone https://github.com/MrKintu/YIkGrzRvc8amq1Sj.git
cd customer-happiness-prediction

Install the required dependencies:
pip install -r requirements.txt

Run the script to train and evaluate the model:
python happiness_prediction.py

Results and Visualizations
The code generates the following outputs:

Visualizations of data distribution and feature relationships.
A trained model with the best hyperparameters from GridSearchCV.
A classification report and confusion matrix showing model performance.
A feature importance plot and RFECV results to identify key questions for future surveys.
License
This project is licensed under the MIT License - see the LICENSE file for details.
