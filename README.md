# Clinical Data Science Pipeline: Diabetes Patient Readmission and Length of Stay Analysis

This repository contains an end-to-end data science pipeline designed to analyze and predict hospital readmission rates and length of stay (time in hospital) for diabetic patients. Using the Diabetes 130-US Hospitals (1999-2008) dataset, the pipeline integrates exploratory data analysis (EDA), preprocessing, clustering, association rule mining, classification, and regression.

The repository is structured as a modular, reproducible Python pipeline where each step runs sequentially, logging detailed metrics and saving professional visualizations.

---

## Dataset Overview

The project uses the **Diabetes 130-US Hospitals Dataset** (often referred to as the Hospital Readmission Dataset) from the UCI Machine Learning Repository. It contains over 100,000 clinical encounters of diabetic patients across 130 US hospitals, spanning 10 years.

* **Raw Dataset Size:** 101,766 encounters across 50 features (demographics, medications, diagnoses, and hospital stay metrics).
* **Clinical Significance:** 
  * **Classification Target:** Predict early readmission (defined as readmission within 30 days) to help hospitals optimize care transition and reduce penalties.
  * **Regression Target:** Predict the patient's length of stay (`time_in_hospital`), which is critical for hospital resource management and scheduling.

---

## Directory Structure

```
├── data/
│   ├── .gitkeep
│   └── diabetic_data.csv                    # Raw dataset (Git-ignored)
├── data_processed/
│   ├── .gitkeep
│   ├── diabetic_preprocessed.csv            # Preprocessed dataset (Git-ignored)
│   └── diabetic_preprocessed_for_rules.csv  # Binarized dataset for Apriori (Git-ignored)
├── logs/                                    # Automated run logs (Git-ignored)
├── outputs/
│   ├── figures/                             # Generated plots and visualization artifacts
│   └── results/                             # Evaluation results and feature importance tables
├── src/
│   ├── 01_explore.py                        # Exploratory Data Analysis & baseline plots
│   ├── 02_preprocess.py                     # Data cleaning, encoding & feature engineering
│   ├── 03_clustering.py                     # K-Means, DBSCAN & Hierarchical clustering
│   ├── 04_association_rules.py              # Association rule mining (Apriori)
│   ├── 05_classification.py                 # Multi-classifier pipeline (Early Readmit)
│   ├── 06_regression.py                     # Multi-regressor pipeline (Length of Stay)
│   └── utils.py                             # Path configurations & helper functions
├── .gitignore                               # Standard Git exclude configuration
└── requirements.txt                         # Project python dependencies
```

---

## Pipeline Stages

### 1. Exploratory Data Analysis (`src/01_explore.py`)
* Analyzes raw data types, value ranges, missing values (marked as `?`), and class distributions.
* Inspects duplicate patients (multi-encounter analysis) and identifies patients who cannot be readmitted (e.g., deceased or transferred to hospice care).
* Saves initial visualizations including target distribution, correlation matrices, and demographic distributions.
* Generates a data quality summary report recommending pipeline preprocessing steps.

### 2. Preprocessing & Feature Engineering (`src/02_preprocess.py`)
* **Cleaning & Filtering:** Converts `?` to null values. Excludes patients who died or went to hospice. Filters out repeat visits, keeping only the first encounter per patient to prevent data leakage during model training.
* **Dimensionality Reduction:** Drops columns with high missing rates (`weight` at 97% and `payer_code` at 40%) or clinical identifiers. Drops medication columns where the prescription rate is near-zero.
* **Feature Encoding:** Maps clinical ICD-9 codes to 17 categorical disease groups. Ordinally encodes age ranges. One-hot encodes nominal categorical columns.
* **Feature Engineering:** Computes engineered clinical features:
  * `total_visits` (outpatient + inpatient + emergency visits)
  * `procedures_per_day` (number of procedures / length of stay)
  * `meds_per_day` (number of medications / length of stay)
  * `lab_per_day` (number of lab procedures / length of stay)
* Creates a binarized version of the dataset optimized for Association Rule Mining.

### 3. Clustering Analysis (`src/03_clustering.py`)
* Evaluates K-Means, Hierarchical (Agglomerative), and DBSCAN clustering on standardized numeric features.
* Projects high-dimensional patient data into 2D spaces using Principal Component Analysis (PCA) for visualization.
* **Out-of-Core Optimization:** Implements a sparse $k$-NN connectivity graph ($k=10$) for Agglomerative Clustering, preventing the memory allocation limits associated with dense distance matrices on large datasets.
* Employs Davies-Bouldin, Calinski-Harabasz, and Silhouette scores to determine optimal cluster configurations and analyzes readmission rates across clusters.

### 4. Association Rule Mining (`src/04_association_rules.py`)
* Extracts frequent itemsets using the Apriori algorithm from the binarized medical profiles.
* Conducts sensitivity analysis measuring how different support thresholds affect rule count.
* Filters and prioritizes rules where the target variable (`early_readmit`) is in the consequent to identify high-risk clinical profiles.
* Visualizes rule metrics (support, confidence, lift) and maps top relationships using lift heatmaps.

### 5. Classification Pipeline (`src/05_classification.py`)
* Evaluates 10 classifier configurations (KNN, Gaussian Naive Bayes, Decision Trees, Random Forests, XGBoost/GBDT, Linear SVM, and Multi-Layer Perceptrons) using **5-fold Stratified Cross-Validation**.
* **Imbalance Handling:** Employs SMOTE for distance/probability classifiers and class-weight balancing for tree and ensemble models to address the severe readmission target skewness.
* Ranks features based on ANOVA F-scores, Mutual Information, and Random Forest importances.
* **Threshold Optimization:** Deep-dives into the best model (by ROC-AUC) and tunes the classification decision threshold to maximize the F1-score.
* **Statistical Validation:** Runs a Wilcoxon signed-rank test on fold-level AUC scores to verify if the best-performing model is statistically superior to the second-best model.

### 6. Regression Pipeline (`src/06_regression.py`)
* Predicts the length of stay (`time_in_hospital`, 1 to 14 days) using 10 regression models, including Linear, Ridge, Lasso, Decision Trees, Random Forests, XGBoost, and Poisson Regressors (tailored for count data).
* **Leakage Guard:** Drops derived features (`lab_per_day`, `meds_per_day`, `procedures_per_day`) because their calculation depends on the target variable, preventing artificial performance inflation.
* Evaluates model sensitivity to target leakage by testing predictions with and without discharge disposition codes.
* Visualizes prediction residuals, fitted values, and feature importances for the best model.
* Utilizes a Wilcoxon signed-rank test on fold-level RMSE values to validate model superiority.

---

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd DataScience
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Add Raw Dataset**
   Download the `diabetic_data.csv` from the [UCI Machine Learning Repository (Diabetes 130-US hospitals dataset)](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008). 
   Create a `data` folder in the project root and place the raw CSV file inside it:
   ```bash
   mkdir -p data
   # Move or copy the downloaded csv to data/diabetic_data.csv
   ```

---

## How to Run the Pipeline

The scripts are numbered in the order they must be executed. Run each script from the project root directory:

```bash
# Step 1: Run Exploratory Data Analysis
python src/01_explore.py

# Step 2: Preprocess the data and engineer features
python src/02_preprocess.py

# Step 3: Run Clustering Algorithms
python src/03_clustering.py

# Step 4: Extract Association Rules
python src/04_association_rules.py

# Step 5: Train and compare Classifiers
python src/05_classification.py

# Step 6: Train and compare Regressors
python src/06_regression.py
```

All metrics, model scores, and execution steps are printed to the console and automatically logged in the `logs/` directory. Final plots and tables are saved in `outputs/figures/` and `outputs/results/` respectively.
