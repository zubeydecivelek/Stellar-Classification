# Stellar Classification Machine Learning Project

## Overview
This machine learning project focuses on classifying stellar types using the Scikit-Learn library. The dataset used is the [Stellar Classification Dataset (SDSS17)](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data), containing 100,000 samples with three class types: galaxy, star, or quasar object.

## Dataset
- **Dataset Link:** [SDSS17 Dataset](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data)
- **Features:**
  - Right Ascension angle, Declination angle, and various photometric system filters.
  - Additional information like run ID, rerun ID, camera column, plate ID, and more.
  - Target variable: Object class (galaxy, star, or quasar).

## Preprocessing
- Checked for missing values (none found).
- Dropped unnecessary columns (e.g., 'obj_ID', 'run_ID', etc.).
- Encoded the 'class' column using LabelEncoder.
- Applied Min-Max scaling to specific features.

## Data Splitting
### Train-Test Split
- Split the dataset into training (80%) and testing (20%) sets.

### 5-fold Cross Validation
- Utilized StratifiedKFold with 5 folds for robust performance estimation.

## Classification Methods
Implemented and evaluated the following classification algorithms:
1. k-Nearest Neighbor (kNN)
2. Weighted k-Nearest Neighbor
3. Naive Bayes
4. Random Forest
5. Support Vector Machines (SVM)

## Metrics
Utilized Scikit-Learn library for evaluation metrics:
- Accuracy, Precision, Recall, F1-Score.

## Implementation Details
- Main script: `stellar_classification.py`.
- Dependencies: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.

## How to Run the Code
1. Clone the repository.
2. Install required dependencies (`pip install numpy pandas matplotlib seaborn scikit-learn`).
3. Run the main script: `python stellar_classification.py`.

## Results and Visualizations
- Presented results for both Train-Test Split and 5-fold Cross-Validation.
- Visualized classification metrics using bar charts.

## Explanations
- Provided insights into the performance of each algorithm.
- Discussed the impact of data distribution on Naive Bayes' performance.
