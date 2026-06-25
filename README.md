[README_Model_SVM.md](https://github.com/user-attachments/files/29340314/README_Model_SVM.md)
# Model SVM

Support Vector Regression project using the red wine quality dataset. The goal is to predict wine quality from physicochemical variables using an SVR model and hyperparameter tuning.

The main analysis is contained in `models.ipynb`, while `data_features.ipynb` performs the data exploration and preprocessing workflow.

## Objective

Build and evaluate a Support Vector Regression model for predicting the `quality` score of red wines.

The project focuses on:

- Preparing the red wine quality dataset.
- Removing duplicate records.
- Reducing correlated variables.
- Removing outliers with a 3-standard-deviation threshold.
- Standardizing the processed features.
- Tuning an SVR model with `GridSearchCV`.
- Evaluating regression performance with MAE, RMSE, MAPE, and R2.

## Repository Structure

```text
Model_SVM_Test/
├── data_features.ipynb
├── models.ipynb
├── __pycache__/
│   └── auxiliary.cpython-311.pyc
└── dataset/
    ├── winequality-red.csv
    └── winequality-red_procesados.csv
```

## Dataset

The repository includes:

- `dataset/winequality-red.csv`: original red wine quality dataset.
- `dataset/winequality-red_procesados.csv`: processed dataset used for modeling.

The original dataset contains physicochemical measurements for red wines and a target quality score.

The processed modeling features are:

- `fixed acidity`
- `volatile acidity`
- `residual sugar`
- `chlorides`
- `total sulfur dioxide`
- `sulphates`
- `alcohol`

The target variable is:

- `quality`: wine quality score.

After preprocessing, the dataset is reduced from 1,359 records after duplicate removal to 1,255 records after outlier pruning.

## Notebooks

### `data_features.ipynb`

This notebook performs the preprocessing workflow:

- Loads the original red wine dataset.
- Reviews dataset structure and summary statistics.
- Removes duplicated records.
- Studies feature distributions.
- Reviews the distribution of the `quality` target.
- Analyzes feature correlations.
- Removes highly correlated features.
- Detects outliers using a 3-standard-deviation threshold.
- Removes rows containing outliers.
- Standardizes the processed features.
- Saves the processed dataset as `winequality-red_procesados.csv`.

### `models.ipynb`

This notebook trains and evaluates an SVR model:

- Loads the processed dataset.
- Uses physicochemical variables as predictors.
- Uses `quality` as the regression target.
- Splits the data into training and test sets.
- Standardizes the input features.
- Tunes SVR hyperparameters with `GridSearchCV`.
- Evaluates the best model on the test set.

## Technologies Used

- Python 3
- Jupyter Notebook
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

## Installation

Clone the repository:

```bash
git clone https://github.com/fabriciolopretto/Model_SVM_Test.git
cd Model_SVM_Test
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
```

## Usage

Start Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in this order:

```text
data_features.ipynb
models.ipynb
```

`data_features.ipynb` creates the processed dataset, and `models.ipynb` trains and evaluates the SVR model.

## Methodology

1. Load `dataset/winequality-red.csv`.
2. Remove duplicate records.
3. Analyze correlations between features.
4. Remove selected highly correlated variables:
   - `citric acid`
   - `density`
   - `free sulfur dioxide`
   - `pH`
5. Detect and remove outliers.
6. Standardize the processed features.
7. Load `dataset/winequality-red_procesados.csv`.
8. Split the dataset into training and test sets with an 80/20 split, stratified by `quality`.
9. Tune an `SVR` model using 5-fold cross-validation.
10. Evaluate the best model on the test set.

## Model

The SVR model is tuned with `GridSearchCV` over:

- Linear kernel with different `C` values.
- RBF and sigmoid kernels with different `C` and `gamma` values.
- Polynomial kernel with different `C` and `degree` values.

The scoring function used during cross-validation is:

```text
negative mean absolute error
```

The best parameters reported in the notebook are:

```text
C: 1
gamma: 0.1
kernel: rbf
```

## Results

The notebook reports the following test-set metrics:

| Metric | Value |
| --- | ---: |
| MAE | `0.4900` |
| RMSE | `0.6585` |
| MAPE | `0.0879` |
| R2 | `0.3086` |

The notebook concludes that the model performance is weak. The positive but low R2 value indicates that the SVR explains only part of the variance in wine quality.

## Notes

This project treats wine quality as a regression target rather than a multiclass classification problem. That choice avoids modeling the target as many discrete classes and allows performance to be evaluated with continuous regression metrics.

A natural extension would be to compare SVR against other regression models such as Linear Regression, Decision Tree Regressor, Random Forest, Gradient Boosting, or XGBoost.

## Reference

Cortez, Paulo, Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Wine Quality. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T

