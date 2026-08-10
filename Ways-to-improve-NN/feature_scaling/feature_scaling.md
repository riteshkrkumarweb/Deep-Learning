# Feature Scaling

**Feature scaling** = converting numerical features to a similar scale so that features with larger values don't dominate the model.

## 1. StandardScaler ⭐

* Most commonly used.
* Centers data around **0** with standard deviation **1**.
* Use for **Logistic Regression, SVM, KNN, PCA, Neural Networks**.
* Good default choice when there are no major outliers.

## 2. RobustScaler ⭐

* Less affected by **outliers**.
* Use when numerical features contain significant outliers.

## 3. Normalization

* Scales each **row/sample** to a unit length.
* Commonly used for **text data, TF-IDF, and vector similarity**.
* Useful when the **direction of the vector** matters more than its magnitude.

## 4. Min-Max Scaling

* Scales features to a **fixed range**, usually **0 to 1**.
* Useful when you want all values within a specific range.
* **Sensitive to outliers**.


