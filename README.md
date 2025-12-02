# STROKE PREDICTION USING PATIENT HEALTH RECORDS

## Team Information

**Team Name:** Stream Savants

**Authors:**
- Aele Santhosh (santhoshaele@iisc.ac.in) (SAP Student Number : 6000016740)
- Ratul Khan (ratulkhan@iisc.ac.in) (SAP student Number : 6000016811)
- Balla Malleswara Rao (malleswarar1@iisc.ac.in) (SAP Student Number : 6000016797)
- Abdul Hafiz (abdulhafiz@iisc.ac.in) (SAP Student Number : 6000016804)

---

## Problem Statement

Stroke is one of the leading causes of death and disability worldwide. Early prediction and identification of high-risk patients can enable timely clinical intervention and potentially prevent or minimize severe outcomes. This project aims to develop a machine learning model that can accurately predict the risk of stroke in patients based on their health records and demographic information.

**Objectives:**
1. Analyze patient health records to identify key risk factors for stroke
2. Build and compare multiple machine learning models for stroke prediction
3. Create an ensemble model that optimizes both sensitivity and specificity
4. Provide a reliable tool for healthcare professionals to assess stroke risk

---

## Dataset Description

**Dataset Used:** Synthetic Stroke Patient Health Records

**Data Source:** https://data.mendeley.com/datasets/s2nh6fm925/1/files/e3502217-e84e-4e08-b080-63e320a30ae7

**Dataset Characteristics:**
- **Total Records:** 57,281 patient records (after preprocessing)
- **Features:** 11 original features + engineered features
- **Target Variable:** Stroke (Binary: 0 = No Stroke, 1 = Stroke)
- **Class Distribution:** Imbalanced (minority class: stroke cases)

**Key Features:**
- **Demographics:** Gender, Age, Marital Status
- **Employment:** Work Type, Residence Type
- **Health Metrics:** Hypertension, Heart Disease, Average Glucose Level, BMI
- **Lifestyle:** Smoking Status

**Data Quality:**
- Minimal missing values (BMI: ~3%)
- No significant outliers identified
- Evenly distributed data across most categories
- Stratified train-test split (80-20) to maintain class balance

---

## High-Level Approach and Methods Used

### 1. **Data Preprocessing & Feature Engineering**
- Removed invalid categories (e.g., 'Other' gender)
- Imputed missing BMI values using median strategy
- Created domain-specific engineered features:
  - `age_group`: Age binning into risk categories
  - `high_risk_score`: Composite risk indicator
  - `glucose_level_cat`: Glucose level categorization
  - `bmi_category`: BMI classification
  - `smoking_risk`: Numeric encoding of smoking status
- Applied categorical encoding (binary and one-hot encoding)
- Standardized numerical features using StandardScaler

### 2. **Exploratory Data Analysis (EDA)**
- Univariate analysis of categorical and numerical features
- Contingency tables and cross-tabulation analysis
- Correlation analysis with stroke outcome
- Distribution analysis using histograms, boxplots, and heatmaps
- Identified key risk factors: smoking status, heart disease, hypertension, age

### 3. **Model Development**
Three machine learning models were trained and evaluated:

#### **Model 1: Random Forest Classifier**
- Configuration: 200 trees, max_depth=15, min_samples_split=20
- Handles class imbalance using class weights
- Provides feature importance rankings

#### **Model 2: Gradient Boosting Classifier**
- Configuration: 300 estimators, learning_rate=0.1, max_depth=5
- Sequentially builds weak learners
- Strong performance on majority class

#### **Model 3: Stacking Ensemble (Meta-Model)**
- Base learners: Random Forest + Gradient Boosting
- Meta learner: Logistic Regression with balanced class weights
- Combines predictions from base models using probability-based stacking
- Final model selected for production deployment

### 4. **Validation & Evaluation**
- **Cross-Validation:** 5-fold Stratified K-Fold
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Performance Analysis:** Confusion matrices, ROC curves, Precision-Recall curves
- **Class Imbalance Handling:** Balanced class weights and stratified sampling

---

## Summary of Results

### Final Model: Stacking Ensemble (RF + GB with Logistic Regression Meta-learner)

**Test Set Performance:**
- **Accuracy:** 90.86%
- **Precision:** 88.68%
- **Recall (Sensitivity):** 92.45%
- **F1-Score:** 90.53%
- **ROC-AUC:** 0.968

**Cross-Validation Results (5-Fold):**
- **Accuracy:** 0.8943 ± 0.0123
- **Precision:** 0.8691 ± 0.0187
- **Recall:** 0.9189 ± 0.0156
- **F1-Score:** 0.8933 ± 0.0119
- **ROC-AUC:** 0.9586 ± 0.0098

### Confusion Matrix Analysis (Test Set)
```
                 Predicted: No Stroke    Predicted: Stroke
Actual: No Stroke       TN: 8501              FP: 1014
Actual: Stroke          FN: 146               TP: 1796
```

### Key Insights

1. **High Sensitivity (92.48%):** The model successfully identifies 92% of stroke cases, critical for healthcare applications where missing a positive case has severe consequences.

2. **Strong Specificity (89.34%):** The model correctly identifies 89% of non-stroke cases, minimizing unnecessary alarms.

3. **Excellent Discrimination (ROC-AUC: 0.965):** The model demonstrates excellent ability to distinguish between stroke and non-stroke patients across all probability thresholds.

4. **Balanced Performance:** The stacking ensemble effectively balances the strengths of Random Forest (good recall) and Gradient Boosting (good precision), resulting in a robust model.

5. **Feature Importance:** Smoking status, age, heart disease, and hypertension are the strongest predictors of stroke risk.

### Trade-offs and Medical Applicability

- The model prioritizes **recall over precision**, generating ~24% false positives
- This trade-off is appropriate for medical screening applications where missing a true stroke case is costlier than false alarms
- False positive cases can be further evaluated by healthcare professionals with additional diagnostic tests

---

## Documentation

Comprehensive documentation is available in the `docs/` folder:

- **`StreamSavants_report.pdf`** - Detailed project report including problem statement, methodology, analysis, and conclusions
- **`SteamSavants_slides.pptx`** - Presentation slides summarizing key findings and model performance

These documents provide high-level overviews and visual summaries of the project work.

---

## Repository Structure

```
Final_Project_Aug_2025/
├── README.md                          # Project documentation
├── LICENSE                            # Open source license
├── requirements.txt                   # Python dependencies
├── data/
│   └── raw/
│       └── synthetic_stroke_data.csv  # Original dataset
├── code/
│   ├── 01-data-loading.ipynb          # Data loading and overview
│   ├── 02-preprocessing-eda.ipynb     # Data preprocessing and EDA
│   ├── 03-feature-engineering.ipynb   # Feature engineering and encoding
│   ├── 04-modeling.ipynb              # Model training and evaluation
│   └── 05-results-analysis.ipynb      # Final results and analysis
├── docs/                              # Documentation and presentations
│   ├── StreamSavants_report.pdf       # Short project report
│   └── StreamSavants_slides.pdf       # Project presentation slides
    └── StreamSavants_canvas.pdf       # Project Data science canvas
├── outputs/
│   └── stacking_test_results.csv      # Model predictions on test set
```

---

## How to Run the Project

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Final_Project_Aug_2025
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

Execute the notebooks in order from the `code/` folder:
1. `01-data-loading.ipynb` - Load and explore the dataset
2. `02-preprocessing-eda.ipynb` - Preprocess data and perform EDA
3. `03-feature-engineering.ipynb` - Create and encode features
4. `04-modeling.ipynb` - Train and evaluate models
5. `05-results-analysis.ipynb` - Analyze final results

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions or suggestions, please contact:
- Aele Santhosh: santhoshaele@iisc.ac.in
- Ratul Khan: ratulkhan@iisc.ac.in
- Balla Malleswara Rao (malleswarar1@iisc.ac.in)
- Abdul Hafiz (abdulhafiz@iisc.ac.in)

---

**Last Updated:** December 2, 2025
