Ordinal Regression on Wine
================
Dhanashree Pokale
4/19/2017

### Aim of this project is to use ordinal regression technique to predict wine quality. The dataset used is from UCI Machine Learning Repository.

### About data:

##### The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine. Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

##### These datasets can be viewed as classification or regression tasks. The classes are ordered and not balanced (e.g. there are munch more normal wines than excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent or poor wines. Also, we are not sure if all input variables are relevant. So it could be interesting to test feature selection methods.

### Attribute Information:

##### For more information, read \[Cortez et al., 2009\].

##### Input variables (based on physicochemical tests):

1.  fixed acidity
2.  volatile acidity
3.  citric acid
4.  residual sugar
5.  chlorides
6.  free sulfur dioxide
7.  total sulfur dioxide
8.  density
9.  pH
10. sulphates
11. alcohol

##### Output variable

1.  quality (score between 0 and 10)

##### Required Packages

-   require(foreign)
-   require(ggplot2)
-   require(MASS)
-   require(lattice)
-   require(survival)
-   require(Formula)
-   require(Hmisc)
-   require(reshape2)

##### We start by clearing the workspace.

``` r
# Load Data
# Clear Workspace####
ls()
```

    ## character(0)

``` r
rm(list = ls())
```

##### Load the data.Here, we shall proceed only with Red Wine Quality Data Analysis.

``` r
# read data ####
red_wine <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")
#white_wine <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";")

# Rename the columns ####
colnames(red_wine) <- c("FixedAcidity","VolatileAcidity","CitriAcid","ResidualSugar","Chlorides","FreeSO2","TotalSO2","Density","pH","Sulphates","Alcohol","Quality")
#colnames(white_wine) <- c("FixedAcidity","VolatileAcidity","CitriAcid","ResidualSugar","Chlorides","FreeSO2","TotalSO2","Density","pH","Sulphates","Alcohol","Quality")
```

##### Analysis for red wine

``` r
# Check data values ####
head(red_wine) 
```

    ##   FixedAcidity VolatileAcidity CitriAcid ResidualSugar Chlorides FreeSO2
    ## 1          7.4            0.70      0.00           1.9     0.076      11
    ## 2          7.8            0.88      0.00           2.6     0.098      25
    ## 3          7.8            0.76      0.04           2.3     0.092      15
    ## 4         11.2            0.28      0.56           1.9     0.075      17
    ## 5          7.4            0.70      0.00           1.9     0.076      11
    ## 6          7.4            0.66      0.00           1.8     0.075      13
    ##   TotalSO2 Density   pH Sulphates Alcohol Quality
    ## 1       34  0.9978 3.51      0.56     9.4       5
    ## 2       67  0.9968 3.20      0.68     9.8       5
    ## 3       54  0.9970 3.26      0.65     9.8       5
    ## 4       60  0.9980 3.16      0.58     9.8       6
    ## 5       34  0.9978 3.51      0.56     9.4       5
    ## 6       40  0.9978 3.51      0.56     9.4       5

##### Quality is the response variable to be classified. 10 represents best, 1 represents worse. This variable has 10 levels.

##### Factorize response variable

``` r
red_wine$Quality <- as.factor(red_wine$Quality)
red_wine$Out <- relevel(red_wine$Quality, ref="3")
```

##### Ordinal Regression For building this model, we will be using the polr command to estimate an ordered logistic regression. Then, we’ll specify Hess=TRUE to let the model output show the observed information matrix from optimization which is used to get standard errors.

##### Sampling variables

##### Set seed as M-number. Sample 75% of data as training data and rest 25% as testing data

``` r
set.seed(5665)
subset_indices = sample(nrow(red_wine),nrow(red_wine)*0.75)
RW_train = red_wine[subset_indices,]
RW_test = red_wine[-subset_indices,]
```

##### For building this model, we will be using the polr command to estimate an ordered logistic regression. Then, we’ll specify Hess=TRUE to let the model output show the observed information matrix from optimization which is used to get standard errors.

``` r
m <- polr(Quality
          ~ Alcohol+FixedAcidity + VolatileAcidity+CitriAcid+ResidualSugar + Chlorides+FreeSO2+TotalSO2+Density+pH +Sulphates, 
          data = RW_train, method = "probit", Hess=TRUE)
summary(m)
```

    ## Call:
    ## polr(formula = Quality ~ Alcohol + FixedAcidity + VolatileAcidity + 
    ##     CitriAcid + ResidualSugar + Chlorides + FreeSO2 + TotalSO2 + 
    ##     Density + pH + Sulphates, data = RW_train, Hess = TRUE, method = "probit")
    ## 
    ## Coefficients:
    ##                      Value Std. Error  t value
    ## Alcohol           0.407108   0.035963  11.3202
    ## FixedAcidity      0.053270   0.032829   1.6227
    ## VolatileAcidity  -1.720681   0.248039  -6.9371
    ## CitriAcid        -0.117184   0.304039  -0.3854
    ## ResidualSugar     0.017737   0.023394   0.7582
    ## Chlorides        -3.063592   0.858204  -3.5698
    ## FreeSO2           0.013439   0.004354   3.0868
    ## TotalSO2         -0.007047   0.001503  -4.6884
    ## Density         -52.835707   0.620780 -85.1118
    ## pH               -0.763636   0.316765  -2.4107
    ## Sulphates         1.829201   0.240837   7.5952
    ## 
    ## Intercepts:
    ##     Value    Std. Error t value 
    ## 3|4 -53.8117   0.6353   -84.7050
    ## 4|5 -52.8187   0.6303   -83.7973
    ## 5|6 -50.7263   0.6279   -80.7808
    ## 6|7 -49.1044   0.6327   -77.6054
    ## 7|8 -47.5781   0.6447   -73.7984
    ## 
    ## Residual Deviance: 2281.324 
    ## AIC: 2313.324

##### Predict on new cases

``` r
head(predict(m, RW_test, type = "p"))
```

    ##               3           4         5         6          7            8
    ## 2  0.0072234565 0.065894227 0.6656176 0.2493956 0.01179299 7.603864e-05
    ## 16 0.0040053638 0.044590755 0.6191830 0.3123142 0.01973620 1.704934e-04
    ## 22 0.0024250846 0.031664297 0.5717865 0.3647770 0.02903027 3.168238e-04
    ## 25 0.0010613431 0.017723872 0.4863605 0.4438077 0.05026069 7.858726e-04
    ## 26 0.0019575599 0.027292971 0.5501989 0.3863516 0.03379298 4.059782e-04
    ## 28 0.0002320963 0.005845363 0.3329185 0.5472275 0.11063849 3.138052e-03

##### Check if the probabilities are statistically different for every scenario or new test case

``` r
exp(cbind(OR=coef(m), confint(m)))
```

    ## Waiting for profiling to be done...

    ##                           OR       2.5 %     97.5 %
    ## Alcohol         1.502467e+00 1.349529786  1.6733531
    ## FixedAcidity    1.054714e+00 0.951762707  1.1687280
    ## VolatileAcidity 1.789442e-01 0.109396807  0.2922314
    ## CitriAcid       8.894216e-01 0.490062536  1.6138398
    ## ResidualSugar   1.017895e+00 0.960049191  1.0791841
    ## Chlorides       4.671960e-02 0.008661284  0.2501894
    ## FreeSO2         1.013529e+00 1.004924071  1.0222213
    ## TotalSO2        9.929782e-01 0.990052957  0.9959041
    ## Density         1.131734e-23          NA         NA
    ## pH              4.659690e-01 0.219393616  0.9895651
    ## Sulphates       6.228909e+00 3.833769088 10.1301126

For this we find out which variables are statistically important. If 0 lies in confidence interval then estimate of coeff is not significant in linear model. Here we are using exponential of 0 which is 1. So, if 1 lies in the confidence interval then the estimate of coeff is not significant.

Here, we see that 1 lies in confidence interval for FixedAcidity, CitricAcid, ResidualSugar. Thus, these coeffs are not significant.

Significant coeffs are - *Alcohol*, *Volatile Acidity*,*Chlorides*,*FreeSO2*, *TotalSO2*, *pH* & *Suphates* **Hence, our model should have only *Alcohol*, *Volatile Acidity*,*Chlorides*,*FreeSO2*, *TotalSO2*, *pH* & *Suphates* Variables.** We, drop other variables and rebuild the model.

OR values here make sense with respect to base case which we have set as 3. \* So, when Alcohol content increases by 1 unit, the odds ratio for Quality of wine increases from 3 to some other level by 1.502. \* when Volatile Acidity increases by 1 unit, the odds ratio for Quality increases from 3 to some other level by 0.17.

##### We rebuild the model using significant variables only.

``` r
m1 <- polr(Quality
          ~ Alcohol + VolatileAcidity + Chlorides+FreeSO2+TotalSO2+pH +Sulphates, 
          data = RW_train, method = "probit", Hess=TRUE)
summary(m1)
```

    ## Call:
    ## polr(formula = Quality ~ Alcohol + VolatileAcidity + Chlorides + 
    ##     FreeSO2 + TotalSO2 + pH + Sulphates, data = RW_train, Hess = TRUE, 
    ##     method = "probit")
    ## 
    ## Coefficients:
    ##                     Value Std. Error t value
    ## Alcohol          0.452583   0.035059  12.909
    ## VolatileAcidity -1.707467   0.206549  -8.267
    ## Chlorides       -3.226053   0.822283  -3.923
    ## FreeSO2          0.014185   0.004285   3.310
    ## TotalSO2        -0.007347   0.001422  -5.167
    ## pH              -0.993187   0.235020  -4.226
    ## Sulphates        1.752372   0.237351   7.383
    ## 
    ## Intercepts:
    ##     Value   Std. Error t value
    ## 3|4 -1.9544  0.8285    -2.3591
    ## 4|5 -0.9603  0.8057    -1.1919
    ## 5|6  1.1352  0.8014     1.4165
    ## 6|7  2.7534  0.8058     3.4172
    ## 7|8  4.2769  0.8167     5.2366
    ## 
    ## Residual Deviance: 2282.982 
    ## AIC: 2306.982

##### Predict on new cases

``` r
head(predict(m1, RW_test, type = "p"))
```

    ##              3           4         5         6          7            8
    ## 2  0.007206619 0.065954271 0.6666795 0.2482820 0.01180064 7.699679e-05
    ## 16 0.004288396 0.046806505 0.6265673 0.3035467 0.01863374 1.573691e-04
    ## 22 0.002107116 0.028807722 0.5592563 0.3773970 0.03205557 3.762897e-04
    ## 25 0.001005356 0.017102530 0.4822926 0.4468990 0.05186351 8.370158e-04
    ## 26 0.001922574 0.027028053 0.5499397 0.3865215 0.03417038 4.177748e-04
    ## 28 0.000237867 0.005973357 0.3367442 0.5446326 0.10931447 3.097522e-03

##### head of RW\_test

``` r
RW_test[1:4,]
```

    ##    FixedAcidity VolatileAcidity CitriAcid ResidualSugar Chlorides FreeSO2
    ## 2           7.8            0.88      0.00           2.6     0.098      25
    ## 16          8.9            0.62      0.19           3.9     0.170      51
    ## 22          7.6            0.39      0.31           2.3     0.082      23
    ## 25          6.9            0.40      0.14           2.4     0.085      21
    ##    TotalSO2 Density   pH Sulphates Alcohol Quality Out
    ## 2        67  0.9968 3.20      0.68     9.8       5   5
    ## 16      148  0.9986 3.17      0.93     9.2       5   5
    ## 22       71  0.9982 3.52      0.65     9.7       5   5
    ## 25       40  0.9968 3.43      0.63     9.7       6   6

##### Misclassification or Confusion Matrix

``` r
Probability_TestDataset_Quality <- predict(m1, RW_test, type = "p")
predictedWineScore_Test <- predict(m1, RW_test)
table(RW_test$Quality, predictedWineScore_Test)
```

    ##    predictedWineScore_Test
    ##       3   4   5   6   7   8
    ##   3   0   0   5   1   0   0
    ##   4   0   0  11   4   0   0
    ##   5   0   0 119  38   3   0
    ##   6   0   0  57 102   3   0
    ##   7   0   0   2  42   9   0
    ##   8   0   0   0   4   0   0

``` r
error <- predictedWineScore_Test - RW_test$Quality
```

    ## Warning in Ops.factor(predictedWineScore_Test, RW_test$Quality): '-' not
    ## meaningful for factors

##### Misclassification Rate

``` r
mean((predictedWineScore_Test) != RW_test$Quality)  # misclassification error
```

    ## [1] 0.425

##### Modified Misclassification Rate. We apply weights to determine the accuracy. For exact prediction weight of 1 is given. For a prediction that is off by 1 rank, we give the weight of 0.6 and for 2 ranking off prediction, we give the ranking of 0.3. This accuracy measure shows that Ordinal Regression is doing a good job at making predictions close enough to the real ranks of wine quality.

``` r
### classification Rate
x1 <- mean((predictedWineScore_Test == RW_test$Quality))
x2 <- mean(0.60*((as.numeric(predictedWineScore_Test)-1) == RW_test$Quality))
x3 <- mean(0.60*((as.numeric(predictedWineScore_Test)+1) == RW_test$Quality))
x4 <- mean(0.30*((as.numeric(predictedWineScore_Test)-2) == RW_test$Quality))
x5 <- mean(0.30*((as.numeric(predictedWineScore_Test)+2) == RW_test$Quality))
1-sum(x1,x2,x3,x4,x5) #misclassification
```

    ## [1] 0.173

``` r
sum(x1,x2,x3,x4,x5)
```

    ## [1] 0.827
