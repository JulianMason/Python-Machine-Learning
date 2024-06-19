# Python-Machine-Learning
Machine Learning Tasks using Python Programming

# Predict Cancer Mortality Rates in US Counties

This project aims to predict cancer mortality rates in various US counties using machine learning techniques. The dataset includes socio-economic characteristics and other related information for specific counties.

## Project Overview

The goal is to develop a regression model that can predict cancer mortality rates in "unseen" US counties. The provided dataset consists of:

- `Training_data.csv`: Training data with various features/predictors.
- `Training_data_targets.csv`: Corresponding target variables for the training set.
- `Test_data_example.csv`: Example test data.
- `Test_data_example_targets.csv`: Example test data targets.

## Features

The dataset includes the following predictors/features:

- avgAnnCount: Mean number of reported cases of cancer diagnosed annually.
- avgDeathsPerYear: Mean number of deaths per year.
- incidenceRate: Mean per capita (100,000) cancer diagoses.
- medIncome: Median income.
- popEst2015: Population estimates for 2015.
- povertyPercent: Percentage of people living in poverty.
- studyPerCap: Per capita spending on studies.
- binnedInc: Income binned by 10k.
- MedianAge: Median age of residents.
- MedianAgeMale: Median age of male residents.
- MedianAgeFemale: Median age of female residents.
- GeoFips: County identifier.

## Usage

1. **Data Preparation**: Clean and preprocess the data.
2. **Model Training**: Train regression models using the provided training data.
3. **Model Evaluation**: Evaluate the models using the example test data.
4. **Inference**: Use the trained model to predict cancer mortality rates on unseen test data.

## Files

- `Task.ipynb`: Jupyter notebook containing the project code, analysis, and model development.
- `Training_data.csv`: Training data file.
- `Training_data_targets.csv`: Training targets file.
- `Test_data_example.csv`: Example test data file.
- `Test_data_example_targets.csv`: Example test targets file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
