# Heart Disease Prediction

This project implements an end-to-end machine learning pipeline to predict the presence of heart disease using clinical patient data. The objective is to apply fundamental machine learning techniques and evaluate their effectiveness on a real-world healthcare dataset.

## Dataset

The dataset is based on the UCI Heart Disease dataset and includes clinical attributes such as age, chest pain type, cholesterol levels, ECG results, and exercise-induced angina. The target variable indicates whether a patient has heart disease.

## Methodology

- Data cleaning and handling of missing values  
- Encoding of categorical variables  
- Stratified train-test split to preserve class distribution  
- Model training using Logistic Regression  
- Model comparison with Decision Tree  
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

## Results

- Logistic Regression achieved approximately **84% accuracy**
- The model showed better recall for heart disease cases compared to Decision Tree
- Logistic Regression was selected as the final model due to its stability and interpretability

## Tools Used

- Python  
- Pandas  
- Scikit-learn  
- Matplotlib  

## Conclusion

This project demonstrates how machine learning techniques can be applied to healthcare data to assist in predicting heart disease risk. The results support Logistic Regression as a strong and reliable baseline model for binary classification tasks in medical datasets.
