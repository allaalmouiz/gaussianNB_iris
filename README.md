# Gaussian Naive Bayes: Iris Classification Challenge

## Overview

This project implements a machine learning solution to predict iris flower species based on physical measurements. It demonstrates the implementation of Gaussian Naive Bayes from scratch and compares it with scikit-learn's built-in version.

## Problem Statement

Given measurements of an iris flower's sepal length, sepal width, petal length, and petal width, predict which species the flower belongs to (Setosa, Versicolor, or Virginica).


<p align="center">
  <img src="GNB.png" alt="GNB Chain of Thoughts" width="900">
</p>

## Objectives

- Build a machine learning classification model to predict iris species
- Implement Gaussian Naive Bayes from scratch
- Compare custom implementation with scikit-learn's version
- Evaluate both models' performance using multiple metrics

## Dataset

The project uses the classic **Iris Dataset** containing 150 samples (149 after removing duplicates) with the following features:

- **sepal length** (cm)
- **sepal width** (cm)
- **petal length** (cm)
- **petal width** (cm)
- **target** - Species classification:
  - 0: Setosa
  - 1: Versicolor
  - 2: Virginica

## Key Findings from EDA

- **No missing values** and **minimal duplicates** (1 duplicate removed)
- **High positive correlation** (~0.95) between petal measurements and iris species
- **Weak correlation** between sepal length and species
- **Clear separation** of Setosa from other species, while Versicolor and Virginica overlap in sepal features

## Project Structure

### 1. Data Preparation
- Import required libraries (NumPy, Pandas, Matplotlib, Seaborn, scikit-learn)
- Load iris dataset from scikit-learn
- Create DataFrame and perform initial exploration
- Clean data (remove duplicates, handle whitespace in column names)

### 2. Exploratory Data Analysis (EDA)
- **Statistical Summary:** Dataset shape, data types, null values
- **Correlation Heatmap:** Identify relationships between features
- **Pair Plot with KDEs:** Visualize feature distributions by species

### 3. Model Development

#### Part 1: Gaussian Naive Bayes from Scratch
Implemented custom `GaussianNB_scratch` class with three methods:

- **fit():** Calculate mean, variance, and prior probabilities for each class
- **_log_gaussian_pdf():** Compute log probability using Gaussian distribution
- **predict():** Classify new samples using log-likelihood and priors

#### Part 2: Scikit-learn Implementation
Used scikit-learn's `GaussianNB` for comparison with the scratch implementation.

### 4. Model Evaluation

Both models achieved identical performance on test data:

- **Accuracy:** 100%
- **Precision:** 1.0 (weighted average)
- **Recall:** 1.0 (weighted average)
- **Confusion Matrix:** Perfect classification with no misclassifications

## Results

### Confusion Matrix (Both Models)
```
[[10  0  0]
 [ 0  9  0]
 [ 0  0 11]]
```

### Classification Report
Perfect performance across all iris species with 100% precision, recall, and F1-score.

### Test Case
Sample input `[4.9, 3.0, 1.4, 0.2]` → Correctly predicted as **Setosa (0)**

## Technologies Used

| Library | Purpose |
|---------|---------|
| **NumPy** | Scientific computing, linear algebra |
| **Pandas** | Data manipulation and analysis |
| **Matplotlib** | Data visualization |
| **Seaborn** | Statistical visualization |
| **scikit-learn** | Machine learning models and metrics |

## Key Insights

1. **No Overfitting:** Both models maintain 50/50 train-test distribution with consistent performance
2. **Pattern Recognition:** Models successfully learn all distinguishing patterns in the data
3. **Implementation Equivalence:** Custom from-scratch implementation produces identical results to scikit-learn
4. **Feature Importance:** Petal measurements are significantly more predictive than sepal measurements

## How to Run

1. Ensure you have Python 3.x installed with required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

2. Run the Notebook or Python script in Google Colab

3. The notebook will:
   - Load and prepare the iris dataset
   - Perform exploratory data analysis
   - Train both Gaussian Naive Bayes models
   - Display evaluation metrics and visualizations

## Files

- `gaussian_naive_bayes_iris_challenge.ipynb` - Complete Jupyter notebook with all code and visualizations
- `README.md` - This file

## Conclusion

Both the custom Gaussian Naive Bayes implementation and scikit-learn's version achieved perfect classification on the Iris dataset. This demonstrates that the from-scratch implementation correctly implements the mathematical foundations of Gaussian Naive Bayes while maintaining consistency with industry-standard libraries.

## Additional Resources

- Mathematical derivation and detailed concept explanation available in the project documentation
- Original dataset: [UCI Machine Learning Repository - Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)

---

**Status:** ✅ Complete  
**Last Updated:** 2026
