
# Multi-Class Classification Project Using Machine Learning and Deep Learning

## Introduction
In this project, we built a multi-class classification model (3 classes) using various machine learning and deep learning techniques on a given dataset. The goal was to classify samples into the correct categories.

---

## Steps Taken

### 1. Data Preparation
- Split the dataset into training, validation, and test sets.
- Used stratified splitting (`train_test_split` with `stratify`) to maintain class distribution.
- Applied feature scaling (StandardScaler) to normalize the features.

### 2. Handling Class Imbalance
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class in the training data to balance the dataset.

### 3. Model Building and Experimentation

#### Logistic Regression
- Trained a baseline logistic regression model.
- Achieved reasonable performance but limited due to model simplicity.

#### Stacking Classifier
- Combined multiple base models (RandomForest, XGBoost, Logistic Regression) using a stacking ensemble.
- Used Logistic Regression as the meta-learner.
- Achieved best overall performance with ~79% validation accuracy and 75% test accuracy.
- Outperformed individual base models.

---

## Final Results

| Model               | Validation Accuracy | Test Accuracy |
|---------------------|---------------------|---------------|
| Logistic Regression  | ~75-79%             | 75%           |
| Stacking Classifier  | 79%                 | 75%           |

**Notes:**
- Class 2 had lower precision and recall compared to classes 0 and 1.
- Various techniques such as SMOTE, Dropout, BatchNormalization, and learning rate reduction were used to improve the model.
- Further improvements possible with enhanced feature engineering or other models.

---

## Future Recommendations
- Experiment with class weights to improve minority class performance.
- Add more diverse models in the stacking ensemble.
- Try deeper or different neural network architectures (CNN, RNN depending on data).
- Use advanced feature engineering or feature selection techniques.

---

## Requirements

- Python 3.x  
- Libraries: scikit-learn, xgboost, tensorflow, imblearn, matplotlib, numpy, pandas

---

## Usage Instructions

1. Prepare and split the data.
2. Apply SMOTE for balancing classes.
3. Scale features.
4. Train different models.
5. Evaluate on validation and test sets.
6. Compare and choose the best performing model.

