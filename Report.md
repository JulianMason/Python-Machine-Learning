# Predict Cancer Mortality Rates in US Counties

## Introduction

The objective of this project is to predict cancer mortality rates in various US counties using machine learning techniques. The dataset comprises socio-economic characteristics and other related information for specific counties.

## Data Description

The provided dataset includes:

- `Training_data.csv`: Contains features/predictors related to socio-economic characteristics.
- `Training_data_targets.csv`: Corresponding target variables (cancer mortality rates) for the training set.
- `Test_data_example.csv` and `Test_data_example_targets.csv`: Example test data to prepare the inference script.

### Data Dictionary

- **avgAnnCount**: Mean number of reported cases of cancer diagnosed annually.
- **avgDeathsPerYear**: Mean number of deaths per year.
- **incidenceRate**: Mean per capita (100,000) cancer diagnoses.
- **medIncome**: Median income.
- **popEst2015**: Population estimates for 2015.
- **povertyPercent**: Percentage of people living in poverty.
- **studyPerCap**: Per capita spending on studies.
- **binnedInc**: Income binned by 10k.
- **MedianAge**: Median age of residents.
- **MedianAgeMale**: Median age of male residents.
- **MedianAgeFemale**: Median age of female residents.
- **GeoFips**: County identifier.

## Methodology

### Data Preparation

- **Cleaning**: Handling missing values, removing duplicates, and correcting data types.
- **Normalization**: Scaling numerical features to ensure they contribute equally to the model.
- **Feature Selection**: Selecting the most relevant features for the model.

### Model Training

Various regression models were trained, including:

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression

The models were evaluated based on their performance on the training data.

### Model Evaluation

The models were evaluated using the example test data provided. Key metrics for evaluation included:

- Mean Squared Error (MSE)
- R-squared (R²) score

### Inference

An inference script was developed to predict cancer mortality rates on unseen test data. The script ensures compatibility with the final test data set that will be used for evaluation.

## Results

The best-performing model was identified based on the evaluation metrics. The final model showed promising results in predicting cancer mortality rates, with a reasonable balance between bias and variance.

## Conclusion

This project successfully developed a regression model to predict cancer mortality rates in US counties. The methodology included data preparation, model training, evaluation, and inference, ensuring a robust approach to the regression task.

## Future Work

Future improvements could include:

- Incorporating more features to improve model accuracy.
- Using advanced techniques like ensemble learning.
- Conducting hyperparameter tuning to optimize model performance.
