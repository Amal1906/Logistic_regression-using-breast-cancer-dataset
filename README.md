# Logistic Regression Implementation

## 1) Loading the Dataset
The Breast Cancer dataset from the `sklearn.datasets` module is used. It contains features extracted from breast cancer cell nuclei, and the target variable indicates whether the cancer is malignant or benign.

## 2) Splitting the Data
The dataset is divided into training (80%) and testing (20%) subsets using `train_test_split()`.

## 3) Training the Logistic Regression Model
- The model initializes weights and bias to zero.
- The gradient descent algorithm updates weights iteratively.
- The sigmoid function is used to compute predicted probabilities.
- Predictions are compared with actual labels to adjust weights using gradient descent.

## 4) Making Predictions
- The linear model output is transformed using the sigmoid function.
- Predictions are classified as 1 if the sigmoid output is greater than 0.5, otherwise 0.

## 5) Evaluating the Model
- The model's performance is evaluated using accuracy:
  
  **Accuracy = (Correct Predictions / Total Predictions) × 100**

## 6) Results and Visualization
- The accuracy of the model is printed.
- Additional evaluation metrics, such as a confusion matrix, can be used to analyze the model’s performance further.

