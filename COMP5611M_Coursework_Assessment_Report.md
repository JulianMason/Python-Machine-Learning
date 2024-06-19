# **Predict Cancer Mortality Rates in US Counties**

The provided dataset comprises data collected from multiple counties in the US. The regression task for this assessment is to predict cancer mortality rates in "unseen" US counties, given some training data. The training data ('Training_data.csv') comprises various features/predictors related to socio-economic characteristics, amongst other types of information for specific counties in the country. The corresponding target variables for the training set are provided in a separate CSV file ('Training_data_targets.csv'). Use the notebooks provided for lab sessions throughout this module to provide solutions to the exercises listed below. Throughout all exercises text describing your code and answering any questions included in the exercise descriptions should be included as part of your submitted solution.

Note - We also provide an example test data set ('Test_data_example.csv' and 'Test_data_example_targets.csv'). This is just an example of the final test set (which will NOT be provided to you) that will be used to evaluate your solutions when your submitted solutions are being marked. The provided Test data (I.e. 'Test_data_example.csv' and 'Test_data_example_targets.csv') is NOT to be used as an independent test set when developing your models, but only to prepare your 'prediction/inference' script to make predictions on completely unseen data. Part of this assessment requires you to write such an inference script that evaluates your best, trained regression model on the final test data set such that, we are able to run the inference script ourselves on the unseen (i.e. data we have not provided to you) test data. Yyou can use the example test data ('Test_data_example.csv' and 'Test_data_example_targets.csv') to verify that it works prior to submission.

The list of predictors/features available in this data set are described below:

**Data Dictionary**

avgAnnCount: Mean number of reported cases of cancer diagnosed annually

avgDeathsPerYear: Mean number of reported mortalities due to cancer

incidenceRate: Mean per capita (100,000) cancer diagoses

medianIncome: Median income per county 

popEst2015: Population of county 

povertyPercent: Percent of populace in poverty 

MedianAge: Median age of county residents 

MedianAgeMale: Median age of male county residents 

MedianAgeFemale: Median age of female county residents 

AvgHouseholdSize: Mean household size of county 

PercentMarried: Percent of county residents who are married 

PctNoHS18_24: Percent of county residents ages 18-24 highest education attained: less than high school 

PctHS18_24: Percent of county residents ages 18-24 highest education attained: high school diploma 

PctSomeCol18_24: Percent of county residents ages 18-24 highest education attained: some college 

PctBachDeg18_24: Percent of county residents ages 18-24 highest education attained: bachelor's degree 

PctHS25_Over: Percent of county residents ages 25 and over highest education attained: high school diploma 

PctBachDeg25_Over: Percent of county residents ages 25 and over highest education attained: bachelor's degree 

PctEmployed16_Over: Percent of county residents ages 16 and over employed 

PctUnemployed16_Over: Percent of county residents ages 16 and over unemployed 

PctPrivateCoverage: Percent of county residents with private health coverage 

PctPrivateCoverageAlone: Percent of county residents with private health coverage alone (no public assistance) 

PctEmpPrivCoverage: Percent of county residents with employee-provided private health coverage 

PctPublicCoverage: Percent of county residents with government-provided health coverage 

PctPubliceCoverageAlone: Percent of county residents with government-provided health coverage alone 

PctWhite: Percent of county residents who identify as White 

PctBlack: Percent of county residents who identify as Black 

PctAsian: Percent of county residents who identify as Asian 

PctOtherRace: Percent of county residents who identify in a category which is not White, Black, or Asian 

PctMarriedHouseholds: Percent of married households 

BirthRate: Number of live births relative to number of women in county 

# **Exercise 1**

Read in the training data and targets files. The training data comprises features/predictors while the targets file comprises the targets (i.e. cancer mortality rates in US counties) you need to train models to predict. Plot histograms of all features to visualise their distributions and identify outliers. Do you notice any unusual values for any of the features? If so comment on these in the text accompanying your code. Compute correlations of all features with the target variable (across the data set) and sort them according the strength of correlations. Which are the top five features with strongest correlations to the targets? Plot these correlations using the scatter matrix plotting function available in pandas and comment on at least two sets of features that show visible correlations to each other. (5 marks)

# **Sample Answer to Exercise 1**

The histograms show that some features have unusual values or outliers. Some plots have significantly taller or shorter bars than the rest of the data while some other plots have unusual distribution patterns, showing multiple peaks -  indicating potential outliers.

| **Plots/features with significantly taller or shorter bars** | **Plots/features with unusual distribution patterns** |
| :-- | :-- |
| avgAnnCount | povertyPercent |
| avgDeathsPerYear | medianAgeMale |
| popEst2015 | medianAgeFemale |
| PctOtherRace | avgHouseholdSize |
| medianAge | PercentMarried |
| PctBlack | PctHS18_24 |
| PctAsian | PctSomeCol18_24 |
| | PctHS25_Over |
| | PctBachDeg25_Over |
| | PctUnemployed16_Over |
| | PctPrivateCoverage |
| | PctPrivateCoverageAlone |
| | PctEmpPrivCoverage |
| | PctPublicCoverage |
| | PctPublicCoverageAlone |


From the scatter matrix, there are some visible correlations between certain pairs of features.

* **povertyPercent and PctPublicCoverageAlone:** Strong positive correlation (0.79)
* **medIncome and povertyPercent:** Strong negative correlation (-0.78)
* **medIncome and PctPublicCoverageAlone:** Strong negative correlation (-0.71)
* **PctBachDeg25_Over and medianIncome:** Strong positive correlation (0.69)


Intuitively, the correlations make sense.
* **povertyPercent and PctPublicCoverageAlone:** the higher the poverty rate, the higher the reliance on only public coverage
* **medIncome and povertyPercent:** the higher the income, the lower the poverty rate
* **medIncome and PctPublicCoverageAlone:** the higher the income, the less likely to rely on only public coverage
* **PctBachDeg25_Over and medianIncome:** the more qualified a county is, the more income they are likely make as a result of better jobs.

# **Exercise 2**

Create an ML pipeline using scikit-learn (as demonstrated in the lab notebooks) to pre-process the training data. (3 marks)

# **Sample Answer to Exercise 2**

# **Exercise 3**

Fit linear regression models to the pre-processed data using: Ordinary least squares (OLS), Lasso and Ridge models. Choose suitable regularisation weights for Lasso and Ridge regression and include a description in text of how they were chosen. In your submitted solution make sure you set the values for the regularisation weights equal to those you identify from your experiment(s). Quantitatively compare your results from all three models and report the best performing one. Include code for all steps above. (10 marks)


# **Sample Answer to Exercise 3**

**Fit all three models**

The regularisation weights (alphas) for Lasso and Ridge regressions are 1 and 1,000 respectively.

They were automatically chosen using a grid search approach to search for the best weight within the specified range of alpha values [0.1, 1, 10, 100, 1000].

scikit-learn's GridSearchCV was used on the Lasso and Ridge models. Using a 5-fold cross-validation method for both models, each alpha value in the range was trained and evaluated multiple times using 5 different subsets of the training data. 

This resulted in the regularisation weights being automatically determined based on their performance in the cross-validation.


**Quantitative comparison of model performance - MSE & R-squared**

The Root Mean Squared Error (MSE) of each model was used to evaluate the best model. The RMSE measures the average magnitude of the difference between the predicted and actual values. The lower the MSE value, the better the model.

By comparing each model's RMSE, the OLS Model is the winning model because it has the lowest RMSE of 18.690. By contrast, Ridge's RMSE is 18.694 while Lasso's RMSE is 18.773.

# **Exercise 4**

Use Lasso regression and the best regularisation weight identified from Exercise 3 to identify the five most important/relevant features for the provided data set and regression task. Report what these are desceding order of their importance. (5 marks)

**The five most important features as shown in Fig. 1 above are:**
* incidenceRate: 10.45
* PctBachDeg25_Over: -7.32
* PctPrivateCoverage: -6.1
* PctMarriedHouseholds: -5.23
* PercentMarried: 4.43

# **Exercise 5**

Fit a Random Forest regression model to the training data and quantitatively evaluate and compare the Random Forest regression model with the best linear regression model identified from Exercise 3. Report which model provides the best results. Next, report the top five most important/relevant features for the provided data set and regression task identified using the Random Forest model. Comment on how these compare with the features identified from Lasso regression? (12 marks)

# **Sample Answers for Exercise 5**

We see that the RMSE for the Random Forest model (18.9) is lower than that of the OLS model (19.6), therefore highlighting that the Random Forest is a better model - Fig. 2

On evaluating with the test data on the Random Forest model, we see that the RMSE here (19.6) is only just higher than the RMSE when cross-validating (18.9), giving confidence in the model because the values of both evaluations are not significantly different.

The output above shows the top 5 features using the Random Forest model

**Random Forest - OLS Comparison**

Fig. 3 above shows that the 5 most important features are different for the Lasso and Random Forest models. However, two features make the top 5 of both models:
* incidenceRate
* PctBachDeg25_Over
    
These features are the top 2 features for both models.

# **Exercise 6**

Use the provided test example data ('Test_data_example.csv' and 'Test_data_example_targets.csv') to write an inference script to evaluate the best regression model identified from preceding exercises. First re-train the chosen regression model using all of the provided training data and test your predictions on the provided example test data. Note - the final evaluation of your submission will be done by replacing this example test data with held out (unseen) test data that is not provided to you. Use the code snippet provided below to prepare your inference script to predict targets for the unseen test data. (3 marks)

# **Sample Answers for Exercise 6**

*   Retrain the best regression model identified with best set of associated hyperparameters on the provided training set (1 mark)
*   Write inference script to accept unseen test data as input similar to the provided example test data, predict targets, and evaluate predictions quantitatively using suitable metrics (2 marks)



# **Classification of 1-year patient mortality following a heart attack**

The provided data set contains data from patients who all suffered heart attacks at some point in the past. Some are still alive and some are not. The data provided contains key clinical information (features) for each patient and the prediction task involves identifying (classifying) which patients are likely to survive for at least one year following the heart attack.
The provided features (clinical variables) to be used as predictors by your classification models include the following:

    1. age-at-heart-attack -- age in years when heart attack occurred
    2. pericardial-effusion -- binary. Pericardial effusion is fluid
			      around the heart.  0=no fluid, 1=fluid
    3. fractional-shortening -- a measure of contracility around the heart
			       lower numbers are increasingly abnormal
    4. epss -- E-point septal separation, another measure of contractility.  
	      Larger numbers are increasingly abnormal.
    5. lvdd -- left ventricular end-diastolic dimension.  This is
	      a measure of the size of the heart at end-diastole.
	      Large hearts tend to be sick hearts.

    6. wall-motion-index -- equals wall-motion-score divided by number of
			   segments seen.  Usually 12-13 segments are seen
			   in an echocardiogram.  
               
The target variable is encoded as a binary outcome of whether a patient survived for 1 year post-heart attack or not. Label '0' indicates that the patient died within one year of a heart attack. Label '1' indicates that the patient survived for at least one year after a heart attack.

# **Exercise 7**

Read in the provided data set for classification of patients at risk of mortality 1-yr post heart attack. Plot histograms of all features to visualise their distributions and identify outliers. Report identified outliters and take steps to deal with outliers (if any) appropriately (3 marks)

# **Exercise 8**

Create a machine learning pipeline using scikit-learn and pre-process the provided data appropriately (3 marks)

# **Exercise 9**

Train logistic regression classifiers, with and without L1 and L2 regularisation, using the provided data and compare and evaluate their performance. Report the best performing classifier, with supporting evidence/justification for why it was identified as the best performing classifier. (14 marks)

By evaluating/predicting with the test data, the model with L1 Regularisation appears to be the best model as shown in the output above. The L2 Regularisation model has a higher accuracy, precision, and F1 score.

To further evaluate the three models in a more consistent way, they are cross-validated below.

**Cross Validation**

By evaluating the performance on test data, the evaluation metrics and confusion matrix show that the Logistic Regression with L1 regularisation has the highest accuracy, precision, and F1 score than the models with L2 and no regularisation. 

However, with cross-validation, L2 has the highest accuracy and precision scores but a lower F1 score than the model with no regularisation.

**L2** is the better model due to its consistency across 5 different subsets of data (cv=5).

# **Exercise 10**

Train a Random Forest classifier using the provided data and quantitatively evaluate and compare the Random Forest classifier  with the best logistic regression classifier identified from Exercise 9. Report which model provides the best results. Next, report the top five most important/relevant features identified using the Random Forest model. (10 marks)

As seen in the Fig. 5 above, the L2 Regularization model performs better than the Random Forest Classifier model. L2 has a higher mean accuracy, precision, and F1 score in their cross-validation

Fig. 6 shows the top 5 important features in the Random Forest Classifier model.

```
/Users/jay/Documents/Documents - Jay's Macbook Pro (13281)/MSc Data Science & Analytics - UoL/Semester 2/Machine Learning/Coursework/Data-for-students-regression/

```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2438 entries, 0 to 2437
Data columns (total 31 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   avgAnnCount              2438 non-null   float64
 1   avgDeathsPerYear         2438 non-null   int64  
 2   incidenceRate            2438 non-null   float64
 3   medIncome                2438 non-null   int64  
 4   popEst2015               2438 non-null   int64  
 5   povertyPercent           2438 non-null   float64
 6   studyPerCap              2438 non-null   float64
 7   MedianAge                2438 non-null   float64
 8   MedianAgeMale            2438 non-null   float64
 9   MedianAgeFemale          2438 non-null   float64
 10  AvgHouseholdSize         2438 non-null   float64
 11  PercentMarried           2438 non-null   float64
 12  PctNoHS18_24             2438 non-null   float64
 13  PctHS18_24               2438 non-null   float64
 14  PctSomeCol18_24          609 non-null    float64
 15  PctBachDeg18_24          2438 non-null   float64
 16  PctHS25_Over             2438 non-null   float64
 17  PctBachDeg25_Over        2438 non-null   float64
 18  PctEmployed16_Over       2319 non-null   float64
 19  PctUnemployed16_Over     2438 non-null   float64
 20  PctPrivateCoverage       2438 non-null   float64
 21  PctPrivateCoverageAlone  1955 non-null   float64
 22  PctEmpPrivCoverage       2438 non-null   float64
 23  PctPublicCoverage        2438 non-null   float64
 24  PctPublicCoverageAlone   2438 non-null   float64
 25  PctWhite                 2438 non-null   float64
 26  PctBlack                 2438 non-null   float64
 27  PctAsian                 2438 non-null   float64
 28  PctOtherRace             2438 non-null   float64
 29  PctMarriedHouseholds     2438 non-null   float64
 30  BirthRate                2438 non-null   float64
dtypes: float64(28), int64(3)
memory usage: 590.6 KB

```

```
<Figure size 1440x1080 with 36 Axes>
```

![Figure](/mnt/data/fig_3.png)

```
1. PctBachDeg25_Over: 0.491
2. incidenceRate: 0.444
3. PctPublicCoverageAlone: 0.440
4. medIncome: 0.417
5. povertyPercent: 0.413

```

```
<Figure size 720x720 with 36 Axes>
```

![Figure](/mnt/data/fig_6.png)

```
avgAnnCount                   0
avgDeathsPerYear              0
incidenceRate                 0
medIncome                     0
popEst2015                    0
povertyPercent                0
studyPerCap                   0
MedianAge                     0
MedianAgeMale                 0
MedianAgeFemale               0
AvgHouseholdSize              0
PercentMarried                0
PctNoHS18_24                  0
PctHS18_24                    0
PctSomeCol18_24            1829
PctBachDeg18_24               0
PctHS25_Over                  0
PctBachDeg25_Over             0
PctEmployed16_Over          119
PctUnemployed16_Over          0
PctPrivateCoverage            0
PctPrivateCoverageAlone     483
PctEmpPrivCoverage            0
PctPublicCoverage             0
PctPublicCoverageAlone        0
PctWhite                      0
PctBlack                      0
PctAsian                      0
PctOtherRace                  0
PctMarriedHouseholds          0
BirthRate                     0
dtype: int64

```

```
Lasso alpha: 0.1
Ridge alpha: 10

```

```
MSE - RMSE: OLS: 349.8020888889968 18.70299678899071
MSE - RMSE: Lasso: 352.59557102496854 18.777528352394214
MSE - RMSE: Ridge: 349.9710787133725 18.70751396400314
R2 - OLS: 0.5749422521069577
R2 - Lasso: 0.5715477863126943
R2 - Ridge: 0.574736906180082

```

```
Winning Model: OLS Model
RMSE: 18.703
R2: 0.575

```

```
1. incidenceRate: 10.45
2. PctBachDeg25_Over: -7.32
3. PctPrivateCoverage: -6.1
4. PctMarriedHouseholds: -5.23
5. PercentMarried: 4.43

```

```
<Figure size 576x432 with 1 Axes>
```

![Figure](/mnt/data/fig_13.png)

```
Scores: [19.0907355  18.51685596 18.81140871 19.20757345 19.07676915]
Mean: 18.940668554397813
Standard Deviation: 0.24843788499814579

```

```
Scores: [20.23846702 18.82990606 19.86712853 18.97830044 20.37389661]
Mean: 19.657539731281737
Standard Deviation: 0.6388919574249242

```

```
<Figure size 432x288 with 1 Axes>
```

![Figure](/mnt/data/fig_17.png)

```
RMSE:  19.73400290222645

```

```
Top Five Features - Random Forest:
PctBachDeg25_Over: 0.212
incidenceRate: 0.197
medIncome: 0.047
PctHS25_Over: 0.044
avgDeathsPerYear: 0.038

```

```
<Figure size 864x432 with 2 Axes>
```

![Figure](/mnt/data/fig_21.png)

```
The MSE of the best model is  502.19
The RMSE of the best model is  22.41
The R-squared of the best model is  0.34

```

```
/Users/jay/Documents/Documents - Jay's Macbook Pro (13281)/MSc Data Science & Analytics - UoL/Semester 2/Machine Learning/Coursework/Data-for-students-classification/

```

```
<Figure size 1080x1080 with 6 Axes>
```

![Figure](/mnt/data/fig_25.png)

```
Logistic Regression without regularization:
Accuracy: 70.37%
Precision: 0.43
F1 Score: 0.43


Logistic Regression with L1 regularization:
Accuracy: 74.07%
Precision: 0.50
F1 Score: 0.46


Logistic Regression with L2 regularization:
Accuracy: 70.37%
Precision: 0.43
F1 Score: 0.43

```

```
Cross-Validation Mean Accuracy Results:
No Regularization Mean Accuracy: 71.24%
L1 Regularization Mean Accuracy: 71.19%
L2 Regularization Mean Accuracy: 72.14%


Cross-Validation F1 Score Results:
No Regularization F1: 0.42
L1 Regularization F1: 0.37
L2 Regularization F1: 0.39


Cross-Validation Precision Results:
No Regularization Precision: 0.51
L1 Regularization Precision: 0.53
L2 Regularization Precision: 0.55

```

```
<Figure size 720x288 with 2 Axes>
```

![Figure](/mnt/data/fig_29.png)

```
Random Forest Classifier:
Accuracy: 70.37%
Precision: 40.00%
F1 Score: 0.33

```

```
Cross-Validation Results:
Random Forest Accuracy: 65.38%
Random Forest Precision: 35.43%
Random Forest F1: 0.33

```

```
<Figure size 432x288 with 1 Axes>
```

![Figure](/mnt/data/fig_33.png)

```
Top five most important features:
                Feature  Importance
2  FractionalShortening    0.213450
3                  epss    0.204625
4                  lvdd    0.203706
1   PericardialEffusion    0.183041
0      AgeAtHeartAttack    0.161179

```

```
<Figure size 576x432 with 1 Axes>
```

![Figure](/mnt/data/fig_36.png)