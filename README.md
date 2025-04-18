# Customer-Churn-Prediction-using-Machine-Learning
## 📊 Telecom Customer Churn Prediction – Machine Learning Project

### 🧠 Objective
The goal of this project is to develop and evaluate machine learning models that can **predict whether a customer will churn** (i.e., leave the telecom service provider) based on historical customer data.

---

### 🗂 Dataset
- **Name:** Telecom Customer Churn Dataset  
- **Source:** Provided (CSV format)  
- **Rows:** 10,000+  
- **Target Variable:** `Churn` (1 = Yes, 0 = No)  
- **Features:** Customer demographics, account information, service usage, and billing details.

---

### 🛠️ Tools & Libraries
- **Python**, **Pandas**, **NumPy**
- **Seaborn**, **Matplotlib** for data visualization
- **Scikit-learn** for preprocessing and ML models
- **XGBoost** for gradient boosting classifier

---

### 🔄 Data Preprocessing
- Dropped missing values to ensure model integrity.
- Categorical features were encoded using `LabelEncoder`.
- Standardized numerical features using `StandardScaler`.
- Data was split into **80% training** and **20% testing**.

---

### ⚙️ Models Used
| Model                 | Description                             |
|----------------------|-----------------------------------------|
| Logistic Regression  | Baseline linear model                   |
| Random Forest        | Ensemble of decision trees              |
| XGBoost              | Gradient boosting model for performance |

---

### 📈 Evaluation Metrics
- **Accuracy**
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix** (Visualized with Seaborn heatmaps)

---

### ✅ Results

| Model               | Accuracy   |
|--------------------|------------|
| Logistic Regression| ~0.80      |
| Random Forest      | ~0.85      |
| XGBoost            | ~0.86      |

> 🔍 *XGBoost performed best overall, capturing complex patterns with slightly higher accuracy and balanced metrics.*

---

### 📌 Key Insights
- Churn prediction can be effectively solved with classification algorithms.
- Ensemble models like Random Forest and XGBoost outperformed simpler linear models.
- Feature encoding and scaling significantly improved model performance.

---

### 📊 Future Improvements
- Perform **feature selection** to reduce overfitting.
- Apply **hyperparameter tuning** (GridSearchCV, Optuna).
- Integrate **Tableau dashboards** for business-level insights.
- Explore **customer segmentation** using unsupervised learning.

---

### 📁 Repository Structure (Suggested)

```
telecom-churn-prediction/
│
├── data/
│   └── telecom_customer_churn.csv
├── notebooks/
│   └── churn_modeling.ipynb
├── models/
│   └── saved_models.pkl
├── outputs/
│   └── confusion_matrices/
├── README.md
├── requirements.txt
└── churn_prediction.py
```
---

### 💼 Relevance to Entry-Level Data Science
This project demonstrates key skills relevant to data science roles:
- Data cleaning and preprocessing
- Model selection and evaluation
- Practical application of classification algorithms
- Visualization and communication of results
