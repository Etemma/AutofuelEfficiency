# Auto MPG: Predicting Vehicle Fuel Efficiency

## Project Overview

This project uses regression techniques to predict vehicle fuel efficiency, measured in miles per gallon (MPG), using vehicle characteristics such as engine displacement, horsepower, weight, cylinders and model year.

The objective was to investigate the factors associated with fuel efficiency and compare different regression models.

## Dataset

- Source: UCI Machine Learning Repository – Auto MPG Dataset
- Observations: 398 vehicles
- Original features: 8 predictors and 1 target variable
- Target variable: "mpg"

### Features

Feature| Description
cylinders| Number of engine cylinders
displacement| Engine displacement
horsepower| Engine horsepower
weight| Vehicle weight
acceleration| Vehicle acceleration
model_year| Vehicle model year
origin| Vehicle origin
car_name| Vehicle name

The "car_name" column was excluded from the initial modelling process because it contains high-cardinality text data.

## Data Cleaning and Preparation

The following steps were performed:

1. Loaded the dataset and assigned appropriate column names.
2. Identified six missing values in the "horsepower" column.
3. Imputed missing horsepower values using comparable vehicles based on characteristics such as manufacturer, cylinders, displacement, weight and model year.
4. Checked for duplicate records.
5. Treated "cylinders" and "origin" as categorical variables.
6. Separated the dataset into features ("X") and target ("y").

The imputed horsepower values were estimates based on comparable observations, not verified factory specifications.

The dataset was divided into training and testing sets using an 80:20 split, with "random_state=42".

- Training observations: 318
- Testing observations: 80

## Exploratory Data Analysis

The exploratory analysis examined the distribution of MPG and relationships between vehicle characteristics and fuel efficiency.

## Key Findings

- Average fuel efficiency was approximately 23.5 MPG.
- Median MPG was 23.
- MPG ranged from 9 to 46.
- The target variable showed a positive skew.
- Vehicle weight had the strongest negative correlation with MPG (approximately -0.83).
- Displacement (-0.804), cylinders (-0.775) and horsepower (-0.774) also had strong negative correlations with MPG.
- Model year showed a positive correlation with MPG (approximately 0.579).

The findings suggest that heavier vehicles and vehicles with larger engines generally have lower fuel efficiency in this dataset.

Correlation does not establish causation.

## Model Development

Three approaches were evaluated.

### 1. Baseline Model

A "DummyRegressor" using the mean strategy was used as the benchmark.

Metric| Result
R²| -0.0056
MAE| 6.645
RMSE| 7.841

The baseline performed poorly, establishing a benchmark for the feature-based models.

### 2. Linear Regression

A preprocessing pipeline was implemented using:

- "ColumnTransformer"
- "OneHotEncoder"
- "Pipeline"
- "LinearRegression"

One-hot encoding was applied to the categorical variables, while the remaining numerical features were passed through unchanged.

Metric| Result
R²| 0.8038
MAE| 2.712
RMSE| 3.463

The model explained approximately 80.38% of the variation in MPG on the test set.

### 3. Random Forest Regression

A Random Forest model was trained using the same preprocessing pipeline.

Metric| Result
R²| 0.867
MAE| 2.118
RMSE| 2.851

The Random Forest outperformed Linear Regression in R² and RMSE, although Linear Regression achieved a slightly lower MAE.

## Model Comparison

Model| R²| MAE| RMSE
Dummy Regressor| -0.0056| 6.645| 7.841
Linear Regression| 0.8038| 2.712| 3.463
Random Forest| 0.867| 2.118| 2.851
Tuned Random Forest| 0.8663| 2.12| 2.859

Model Tuning and Overfitting

The Random Forest model was tuned by adjusting:

- "n_estimators=100"
- "max_depth=10"
- "min_samples_split=10"
- "min_samples_leaf=5"

The original model had a training R² of 0.981 and a testing R² of 0.867.

After tuning:

- Training R²: 0.934
- Testing R²: 0.8663

The difference between training and testing performance decreased from approximately 0.114 to 0.0677.

This indicates reduced overfitting, although the tuned model did not improve test-set performance.

Feature Importance

The Random Forest feature importance analysis identified the following as the most influential features:

1. Displacement
2. Number of cylinders (particularly the four-cylinder category)
3. Model year
4. Horsepower

Other features had comparatively low importance in this model.

Feature importance indicates how much the model relies on a variable; it does not establish a causal relationship.

## Residual Analysis

Residuals were calculated as:

Residual = Actual MPG − Predicted MPG

The tuned model produced:

- Mean residual: approximately -0.55
- Median residual: approximately -0.39
- Residual standard deviation: approximately 2.8
- Minimum residual: -7.94
- Maximum residual: 8.489

The mean residual suggests slight overprediction on average, while the range indicates that some individual predictions still have substantial errors.

Conclusion

Random Forest Regression produced the strongest overall test-set performance, achieving an R² of approximately 0.867 and an RMSE of 2.851.

The project demonstrated the importance of data cleaning, appropriate treatment of categorical variables, preprocessing pipelines, baseline comparisons, model evaluation and hyperparameter tuning.

Although tuning reduced the difference between training and testing performance, it did not materially improve predictive performance. The original Random Forest therefore remained the best-performing model based on the reported test metrics.

## Skills and Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Exploratory Data Analysis
- Missing-value imputation
- Categorical encoding
- Regression modelling
- Random Forest
- Model evaluation
- Hyperparameter tuning
- Residual analysis

## Future Improvements

- Investigate manufacturer-level feature engineering from "car_name".
- Explore cross-validation for more reliable model comparison.
- Investigate feature interactions and multicollinearity.
- Evaluate additional regression algorithms.
