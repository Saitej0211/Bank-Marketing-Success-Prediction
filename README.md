
# Bank Marketing Success Prediction

## Overview

This project aims to predict the success of marketing campaigns conducted by a Portuguese banking institution. The bank's marketing strategy was focused on phone calls, where potential clients were contacted to subscribe to a bank term deposit. Using a dataset with customer information, we applied various machine learning models to classify whether a customer will subscribe to the term deposit.

The project involves:
- Data preprocessing, including feature engineering and handling missing values.
- Exploratory Data Analysis (EDA) to understand data structure and relationships.
- Implementation of classification algorithms such as Logistic Regression and Gaussian Naive Bayes.
- Model evaluation using metrics like accuracy, F1-score, precision, and recall.

## Dataset
The Dataset is taken from https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset consists of 45,211 rows and 17 attributes, representing customer and marketing interaction details. It includes both numerical and categorical features such as:
- **Numerical Features**: `age`, `balance`, `campaign`, `pdays`, `previous`, etc.
- **Categorical Features**: `job`, `education`, `default`, `housing`, `loan`, etc.

The target variable is **deposit**, which indicates whether a customer subscribed to a term deposit (binary outcome: 'yes' or 'no').

## Key Steps

### 1. Data Preprocessing
- **Handling Missing Values**: Missing values in categorical columns such as `poutcome` and `contact` were addressed by either dropping features or imputing values.
- **Outlier Removal**: Outliers in columns such as `campaign` and `previous` were identified and removed to avoid model distortion.
- **Feature Engineering**: Categorical variables were converted to numerical representations using techniques like one-hot encoding and target encoding. Continuous variables like `age` were binned for better analysis.

### 2. Exploratory Data Analysis (EDA)
- Visualization of variable distributions, correlations, and customer demographics.
- Analysis of target variable imbalance and relationships between features and customer behavior (e.g., housing loans, default status, etc.).

### 3. Modeling
We used two primary classification algorithms:
- **Logistic Regression**: Both standard Logistic Regression and PCA-enhanced versions were implemented.
- **Gaussian Naive Bayes**: Assumes feature independence and works well with Gaussian-distributed numerical data.

### 4. Model Evaluation
Models were evaluated using the following metrics:
- **Accuracy**
- **F1-score**
- **Precision**
- **Recall**

### Key Findings
- Logistic Regression performed best with an accuracy of **84.5%** and an F1-score of **0.56**.
- Models without PCA generally outperformed PCA-enhanced models, likely due to the loss of important features during dimensionality reduction.
- Gaussian Naive Bayes performed poorly compared to Logistic Regression, mainly due to the assumption of feature independence.

## Installation and Usage

### Prerequisites
- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `smote`, `jupyter`, etc.

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Saitej0211/Bank-Marketing-Success-Prediction.git
   cd bank-marketing-success-prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebooks to explore data and train models:
   ```bash
   jupyter notebook
   ```

4. To train models directly, run the script:
   ```bash
   python scripts/train_model.py
   ```

## Results and Insights
- **Logistic Regression** showed the highest accuracy, making it the most reliable model for this dataset.
- Customers with housing loans and existing loans are less likely to subscribe to term deposits.
- Seasonal trends affect customer decisions, with peak subscription activity in May and June.
  
## Future Work
- Experiment with more advanced machine learning algorithms like Random Forest, SVM, and XGBoost.
- Further fine-tuning of models with hyperparameter optimization.
- Explore methods like SMOTE and ADASYN for better handling of class imbalances.

## Contributors
- Keerthana Balswamy
- Saiteja Reddy Gajula
- Suja Ganesh Murugan
