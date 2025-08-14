# Loan Prediction 🏦

## Overview
This project predicts whether a loan will be approved for an applicant based on demographic, financial, and loan-related features. It demonstrates **data preprocessing, feature engineering, model building, and evaluation**, showcasing practical **data science skills** applicable to the financial sector.

---

## Dataset
- **Source:** Public loan dataset  
- **Entries:** 614  
- **Features (13):**
  - `Loan_ID` – Unique ID for each loan  
  - `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed` – Applicant details  
  - `ApplicantIncome`, `CoapplicantIncome` – Financial info  
  - `LoanAmount`, `Loan_Amount_Term` – Loan specifics  
  - `Credit_History` – Creditworthiness indicator  
  - `Property_Area` – Urban/Semiurban/Rural  
  - `Loan_Status` – Target variable (Y/N)

---

## Key Steps

### 1. Data Cleaning
- Filled missing values:
  - Categorical → mode  
  - Numerical → median  
- Standardized `Dependents` (`3+` → 3)  
- Encoded categorical variables for modeling  

### 2. Feature Engineering
- Created `Total_Income = ApplicantIncome + CoapplicantIncome`  
- Log-transformed `LoanAmount` and `Total_Income` to reduce skew  
- Optional: `Income_to_Loan_Ratio = Total_Income / LoanAmount`  

### 3. Model Building & Evaluation
- Split data: 80% train / 20% test  
- Models:
  - Logistic Regression (baseline)  
  - Random Forest & XGBoost (advanced)  
- Evaluation Metrics:
  - Accuracy  
  - F1-score  
  - Confusion Matrix  
  - ROC-AUC  

### 4. Insights
- Most important features affecting loan approval:
  - Credit history  
  - Total income  
  - Property area  
  - Education level  

### 5. Visualization
- Distribution plots for numeric features  
- Categorical features vs. loan status  
- Correlation heatmap for feature relationships  

---

## Tech Stack
- Python, Jupyter Notebook  
- Libraries: Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  

---

## Results
- Models achieve **high accuracy** in predicting loan approvals.  
- Visualizations provide insights into patterns between applicant attributes and loan outcomes.  
- Demonstrates ability to handle **real-world financial datasets**, preprocess data, and deploy predictive models.

