import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Read the CSV file using pandas
df = pd.read_csv("star_classification.csv")

"""Pre-processing"""

df.columns

df.dtypes

# Columns are mostly numeric, we will encode "class" column

df.info()

# There is no missing value as seen, so no need to fill null values.

df = df.drop(['obj_ID', 'run_ID', 'rerun_ID', 'field_ID', 'spec_obj_ID', 'fiber_ID'], axis=1)

df.columns

# Encoding categorical data ('class')
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

df.head(10)

# applying scaling, values will be scaled between 0-1
min_max_scaler = MinMaxScaler()

cols_to_scale = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'cam_col', 'redshift', 'plate', 'MJD']

df[cols_to_scale] = min_max_scaler.fit_transform(df[cols_to_scale])

correlation_matrix = df.corr()

plt.figure(figsize=(10, 8))

# Create the heatmap for correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Matrix Heatmap")
plt.xticks(rotation=45)
plt.show()

"""Split the Dataset

### Option.1 : Train-Test Split
"""

X = df.drop(columns=["class"])  # exclude the target variable 'class'
y = df["class"] # target variable 'class'

# split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""### Option.2 : k-Fold Cross Validation (5-folds)"""

# Create a 5-fold cross-validation splitter
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# models will be evaluated with code below
# scores = cross_val_score(model, X, y, cv=kf)

def calculate_metrics(y_test, y_pred):
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='weighted')
  recall = recall_score(y_test, y_pred, average='weighted')
  f1 = f1_score(y_test, y_pred, average='weighted')

  return accuracy, precision, recall, f1

def calculate_mean_metrics(accuracies, precisions, recalls, f1s):
  mean_accuracy = sum(accuracies) / len(accuracies)
  mean_precision = sum(precisions) / len(precisions)
  mean_recall = sum(recalls) / len(recalls)
  mean_f1 = sum(f1s) / len(f1s)

  return mean_accuracy, mean_precision, mean_recall, mean_f1

"""# Classification Methods

## 1. with Train-Test Split
"""

# kNN
#create the knn model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# predict on the test set
y_pred_knn = knn.predict(X_test)

accuracy_knn, precision_knn, recall_knn, f1_knn = calculate_metrics(y_test, y_pred_knn)

# Weighted k-Nearest Neighbors (kNN)
# create the weighted kNN model
weighted_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
weighted_knn.fit(X_train, y_train)

# predict on the test set
y_pred_weighted_knn = weighted_knn.predict(X_test)

accuracy_weighted_knn, precision_weighted_knn, recall_weighted_knn, f1_weighted_knn = calculate_metrics(y_test, y_pred_weighted_knn)

# Naive Bayes
# create the naive bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)

# predict on the test set
y_pred_nb = nb.predict(X_test)

accuracy_nb, precision_nb, recall_nb, f1_nb = calculate_metrics(y_test, y_pred_nb)

# Random Forest
# Create the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = rf.predict(X_test)

# Calculate Random Forest metrics
accuracy_rf, precision_rf, recall_rf, f1_rf = calculate_metrics(y_test, y_pred_rf)

# Support Vector Machines (SVM)
# Create the SVM model
svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

# Predict on the test set
y_pred_svm = svm.predict(X_test)

# Calculate SVM metrics
accuracy_svm, precision_svm, recall_svm, f1_svm = calculate_metrics(y_test, y_pred_svm)

print("Classification Results with Train-Test Split")
print("\t \t kNN \t \t Weighted kNN \t Naive Bayes \t Random Forest \t SVM")
print(f"Accuracy: \t {accuracy_knn:.4f} \t {accuracy_weighted_knn:.4f} \t {accuracy_nb:.4f} \t {accuracy_rf:.4f} \t {accuracy_svm:.4f}")
print(f"Precision: \t {precision_knn:.4f} \t {precision_weighted_knn:.4f} \t {precision_nb:.4f} \t {precision_rf:.4f} \t {precision_svm:.4f}")
print(f"Recall: \t {recall_knn:.4f} \t {recall_weighted_knn:.4f} \t {recall_nb:.4f} \t {recall_rf:.4f} \t {recall_svm:.4f}")
print(f"F1-Score: \t {f1_knn:.4f} \t {f1_weighted_knn:.4f} \t {f1_nb:.4f} \t {f1_rf:.4f} \t {f1_svm:.4f}")


"""## 2. 5-fold Cross-validation"""

# Create and train kNN model
knn = KNeighborsClassifier(n_neighbors=5)

# lists for knn
accuracies_knn = []
precisions_knn = []
recalls_knn = []
f1_scores_knn = []

# perform cross validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    accuracy_fold, precision_fold, recall_fold, f1_fold = calculate_metrics(y_test, y_pred_knn)

    accuracies_knn.append(accuracy_fold)
    precisions_knn.append(precision_fold)
    recalls_knn.append(recall_fold)
    f1_scores_knn.append(f1_fold)

mean_acc_knn, mean_prec_knn, mean_rec_knn, mean_f1_knn = calculate_mean_metrics(accuracies_knn, precisions_knn, recalls_knn, f1_scores_knn)

# Create and train weighted kNN model
weighted_knn = KNeighborsClassifier(n_neighbors=5, weights='distance')

# lists for weighted kNN
accuracies_weighted_knn = []
precisions_weighted_knn = []
recalls_weighted_knn = []
f1_scores_weighted_knn = []

# perform cross-validation for weighted kNN
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    weighted_knn.fit(X_train, y_train)
    y_pred_weighted_knn = weighted_knn.predict(X_test)

    accuracy_fold, precision_fold, recall_fold, f1_fold = calculate_metrics(y_test, y_pred_weighted_knn)

    accuracies_weighted_knn.append(accuracy_fold)
    precisions_weighted_knn.append(precision_fold)
    recalls_weighted_knn.append(recall_fold)
    f1_scores_weighted_knn.append(f1_fold)

mean_acc_weighted_knn, mean_prec_weighted_knn, mean_rec_weighted_knn, mean_f1_weighted_knn = calculate_mean_metrics(accuracies_weighted_knn, precisions_weighted_knn, recalls_weighted_knn, f1_scores_weighted_knn)

# Create and train Naive Bayes model
nb = GaussianNB()

# lists for nb
accuracies_nb = []
precisions_nb = []
recalls_nb = []
f1_scores_nb = []

# perform cross validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    accuracy_fold, precision_fold, recall_fold, f1_fold = calculate_metrics(y_test, y_pred_knn)

    accuracies_nb.append(accuracy_fold)
    precisions_nb.append(precision_fold)
    recalls_nb.append(recall_fold)
    f1_scores_nb.append(f1_fold)

mean_acc_nb, mean_prec_nb, mean_rec_nb, mean_f1_nb = calculate_mean_metrics(accuracies_nb, precisions_nb, recalls_nb, f1_scores_nb)

# Create the Random Forest model

rf = RandomForestClassifier(n_estimators=100, random_state=42)

accuracies_rf = []
precisions_rf = []
recalls_rf = []
f1_scores_rf = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    accuracy_fold, precision_fold, recall_fold, f1_fold = calculate_metrics(y_test, y_pred_rf)

    accuracies_rf.append(accuracy_fold)
    precisions_rf.append(precision_fold)
    recalls_rf.append(recall_fold)
    f1_scores_rf.append(f1_fold)

mean_acc_rf, mean_prec_rf, mean_rec_rf, mean_f1_rf = calculate_mean_metrics(accuracies_rf, precisions_rf, recalls_rf, f1_scores_rf)

# Support Vector Machines (SVM) within 5-fold cross-validation

accuracies_svm = []
precisions_svm = []
recalls_svm = []
f1_scores_svm = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    svm = SVC(kernel='linear', C=1.0, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)

    accuracy_fold, precision_fold, recall_fold, f1_fold = calculate_metrics(y_test, y_pred_svm)

    accuracies_svm.append(accuracy_fold)
    precisions_svm.append(precision_fold)
    recalls_svm.append(recall_fold)
    f1_scores_svm.append(f1_fold)

mean_acc_svm, mean_prec_svm, mean_rec_svm, mean_f1_svm = calculate_mean_metrics(accuracies_svm, precisions_svm, recalls_svm, f1_scores_svm)

print("Classification Results with 5-fold Cross-Validation")
print("\t \t kNN \t \t Weighted kNN \t Naive Bayes \t Random Forest \t SVM")
print(f"Accuracy: \t {mean_acc_knn:.4f} \t {mean_acc_weighted_knn:.4f} \t {mean_acc_nb:.4f} \t {mean_acc_rf:.4f} \t {mean_acc_svm:.4f}")
print(f"Precision: \t {mean_prec_knn:.4f} \t {mean_prec_weighted_knn:.4f} \t {mean_prec_nb:.4f} \t {mean_prec_rf:.4f} \t {mean_prec_svm:.4f}")
print(f"Recall: \t {mean_rec_knn:.4f} \t {mean_rec_weighted_knn:.4f} \t {mean_rec_nb:.4f} \t {mean_rec_rf:.4f} \t {mean_rec_svm:.4f}")
print(f"F1-Score: \t {mean_f1_knn:.4f} \t {mean_f1_weighted_knn:.4f} \t {mean_f1_nb:.4f} \t {mean_f1_rf:.4f} \t {mean_f1_svm:.4f}")

"""# Explanations and Plots"""

# Scores of train-test split
models = ['kNN', 'Weighted kNN', 'Naive Bayes', 'Random Forest', 'SVM']
accuracy = [accuracy_knn, accuracy_weighted_knn, accuracy_nb, accuracy_rf, accuracy_svm]
precision = [precision_knn, precision_weighted_knn, precision_nb, precision_rf, precision_svm]
recall = [recall_knn, recall_weighted_knn, recall_nb, recall_rf, recall_svm]
f1_score = [f1_knn, f1_weighted_knn, f1_nb, f1_rf, f1_svm]

bar_width = 0.2
x = range(len(models))
plt.figure(figsize=(10, 6))

plt.bar(x, accuracy, width=bar_width, label='Accuracy', align='center', alpha=0.7)
plt.bar([i + bar_width for i in x], precision, width=bar_width, label='Precision', align='center', alpha=0.7)
plt.bar([i + 2*bar_width for i in x], recall, width=bar_width, label='Recall', align='center', alpha=0.7)
plt.bar([i + 3*bar_width for i in x], f1_score, width=bar_width, label='F1-Score', align='center', alpha=0.7)

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Classification Results with Train-Test Split')
plt.xticks([i + 1.5 * bar_width for i in x], models)
plt.legend()
plt.show()

# Scores of 5-fold cross validation
models = ['kNN', 'Weighted kNN', 'Naive Bayes', 'Random Forest', 'SVM']
accuracy = [mean_acc_knn, mean_acc_weighted_knn, mean_acc_nb, mean_acc_rf, mean_acc_svm]
precision = [mean_prec_knn, mean_prec_weighted_knn, mean_prec_nb, mean_prec_rf, mean_prec_svm]
recall = [mean_rec_knn, mean_rec_weighted_knn, mean_rec_nb, mean_rec_rf, mean_rec_svm]
f1_score = [mean_f1_knn, mean_f1_weighted_knn, mean_f1_nb, mean_f1_rf, mean_f1_svm]

bar_width = 0.2
x = range(len(models))
plt.figure(figsize=(10, 6))

plt.bar(x, accuracy, width=bar_width, label='Accuracy', align='center', alpha=0.7)
plt.bar([i + bar_width for i in x], precision, width=bar_width, label='Precision', align='center', alpha=0.7)
plt.bar([i + 2*bar_width for i in x], recall, width=bar_width, label='Recall', align='center', alpha=0.7)
plt.bar([i + 3*bar_width for i in x], f1_score, width=bar_width, label='F1-Score', align='center', alpha=0.7)

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Classification Results with 5-fold Cross-Validation')
plt.xticks([i + 1.5 * bar_width for i in x], models)
plt.legend()
plt.show()


""" 
    In the 5-fold Cross-Validation results, we can see that Random Forest achieved the highest Accuracy, Precision, Recall, and F1-Score. 
    On the other hand, Naive Bayes, kNN and weighted-kNN performed consistently but slightly lower in terms of metrics. 
    SVM also performed well, but not as high as Random Forest.

    In the Train-Test Split results, Random Forest still performs exceptionally well with high Accuracy, Precision, Recall, and F1-Score. 
    SVM also remains competitive. kNN and weighted-kNN performed consistently. However, the performance of Naive Bayes dropped noticeably.


    ### 5-fold Cross-Validation:

    Random Forest is the top performer in the 5-fold Cross-Validation due to its ability to handle complex data relationships. 
    Weighted kNN, an extended version of kNN, also performs well. 
    Naive Bayes and kNN, relying on probabilistic concepts, outperform each other but have slightly lower performance. 
    SVM, seeking the optimal hyperplane for class separation, remains a competitive choice for the dataset.


    ### Train - Test Split

    In the Train-Test Split scenario, Random Forest and kNN consistently performs well, even with variations in data distribution. 
    Weighted kNN, an enhanced version of kNN, also performs well, suggesting that assigning different weights to nearest neighbors can improve model performance. 
    Naive Bayes is competitive but sensitive to data distribution shifts, causing inconsistent performance across evaluation metrics. 
    SVM is a reliable choice, consistently delivering competitive results by identifying an optimal hyperplane for class separation.
"""

