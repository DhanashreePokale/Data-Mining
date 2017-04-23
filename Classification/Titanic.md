Kaggle Competition: Titanic: Machine Learning from Disaster
================
Dhanashree Pokale
4/22/2017

Kaggle Competition: Titanic: Machine Learning from Disaster

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

Data Dictionary

Key 0 = No, 1 = Yes 1 = 1st, 2 = 2nd, 3 = 3rd

Variable & its Definition
survival - Survival
pclass - Ticket class
sex - Sex Age - Age in years
sibsp - \# of siblings / spouses aboard the Titanic
parch - \# of parents / children aboard the Titanic
ticket - Ticket number
fare - Passenger fare
cabin - Cabin number
embarked - Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

Variable Notes pclass: A proxy for socio-economic status (SES) 1st = Upper 2nd = Middle 3rd = Lower

age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

sibsp: The dataset defines family relations in this way... Sibling = brother, sister, stepbrother, stepsister Spouse = husband, wife (mistresses and fiancés were ignored)

parch: The dataset defines family relations in this way... Parent = mother, father Child = daughter, son, stepdaughter, stepson Some children travelled only with a nanny, therefore parch=0 for them.

##### Set working directory

``` r
setwd("/Users/dhanashreepokale/Downloads/Independent Projects/Titanic")
```

#### Read the training dataset

Read the file. We don't want to file to be read as factors, hence we specify stringsAsFactors= FALSE

``` r
train_data <- read.csv(file = "train.csv", stringsAsFactors = FALSE, header = TRUE)
test_data <- read.csv(file = "test.csv", stringsAsFactors = FALSE, header = TRUE)
```

#### Verify if the data loading was proper

``` r
tail(train_data)
```

    ##     PassengerId Survived Pclass                                     Name
    ## 886         886        0      3     Rice, Mrs. William (Margaret Norton)
    ## 887         887        0      2                    Montvila, Rev. Juozas
    ## 888         888        1      1             Graham, Miss. Margaret Edith
    ## 889         889        0      3 Johnston, Miss. Catherine Helen "Carrie"
    ## 890         890        1      1                    Behr, Mr. Karl Howell
    ## 891         891        0      3                      Dooley, Mr. Patrick
    ##        Sex Age SibSp Parch     Ticket   Fare Cabin Embarked
    ## 886 female  39     0     5     382652 29.125              Q
    ## 887   male  27     0     0     211536 13.000              S
    ## 888 female  19     0     0     112053 30.000   B42        S
    ## 889 female  NA     1     2 W./C. 6607 23.450              S
    ## 890   male  26     0     0     111369 30.000  C148        C
    ## 891   male  32     0     0     370376  7.750              Q

``` r
tail(test_data)
```

    ##     PassengerId Pclass                           Name    Sex  Age SibSp
    ## 413        1304      3 Henriksson, Miss. Jenny Lovisa female 28.0     0
    ## 414        1305      3             Spector, Mr. Woolf   male   NA     0
    ## 415        1306      1   Oliva y Ocana, Dona. Fermina female 39.0     0
    ## 416        1307      3   Saether, Mr. Simon Sivertsen   male 38.5     0
    ## 417        1308      3            Ware, Mr. Frederick   male   NA     0
    ## 418        1309      3       Peter, Master. Michael J   male   NA     1
    ##     Parch             Ticket     Fare Cabin Embarked
    ## 413     0             347086   7.7750              S
    ## 414     0          A.5. 3236   8.0500              S
    ## 415     0           PC 17758 108.9000  C105        C
    ## 416     0 SOTON/O.Q. 3101262   7.2500              S
    ## 417     0             359309   8.0500              S
    ## 418     1               2668  22.3583              C

#### Check structure of datasets

``` r
str(train_data)
```

    ## 'data.frame':    891 obs. of  12 variables:
    ##  $ PassengerId: int  1 2 3 4 5 6 7 8 9 10 ...
    ##  $ Survived   : int  0 1 1 1 0 0 0 0 1 1 ...
    ##  $ Pclass     : int  3 1 3 1 3 3 1 3 3 2 ...
    ##  $ Name       : chr  "Braund, Mr. Owen Harris" "Cumings, Mrs. John Bradley (Florence Briggs Thayer)" "Heikkinen, Miss. Laina" "Futrelle, Mrs. Jacques Heath (Lily May Peel)" ...
    ##  $ Sex        : chr  "male" "female" "female" "female" ...
    ##  $ Age        : num  22 38 26 35 35 NA 54 2 27 14 ...
    ##  $ SibSp      : int  1 1 0 1 0 0 0 3 0 1 ...
    ##  $ Parch      : int  0 0 0 0 0 0 0 1 2 0 ...
    ##  $ Ticket     : chr  "A/5 21171" "PC 17599" "STON/O2. 3101282" "113803" ...
    ##  $ Fare       : num  7.25 71.28 7.92 53.1 8.05 ...
    ##  $ Cabin      : chr  "" "C85" "" "C123" ...
    ##  $ Embarked   : chr  "S" "C" "S" "S" ...

``` r
str(test_data)
```

    ## 'data.frame':    418 obs. of  11 variables:
    ##  $ PassengerId: int  892 893 894 895 896 897 898 899 900 901 ...
    ##  $ Pclass     : int  3 3 2 3 3 3 3 2 3 3 ...
    ##  $ Name       : chr  "Kelly, Mr. James" "Wilkes, Mrs. James (Ellen Needs)" "Myles, Mr. Thomas Francis" "Wirz, Mr. Albert" ...
    ##  $ Sex        : chr  "male" "female" "male" "male" ...
    ##  $ Age        : num  34.5 47 62 27 22 14 30 26 18 21 ...
    ##  $ SibSp      : int  0 1 0 0 1 0 0 1 0 2 ...
    ##  $ Parch      : int  0 0 0 0 1 0 0 1 0 0 ...
    ##  $ Ticket     : chr  "330911" "363272" "240276" "315154" ...
    ##  $ Fare       : num  7.83 7 9.69 8.66 12.29 ...
    ##  $ Cabin      : chr  "" "" "" "" ...
    ##  $ Embarked   : chr  "Q" "S" "Q" "S" ...

I see that one variable "survived" is missing in test\_data. Hence, lets create a new survived column in test set and fill it with "NA".

``` r
test_data$Survived <- NA
head(test_data)
```

    ##   PassengerId Pclass                                         Name    Sex
    ## 1         892      3                             Kelly, Mr. James   male
    ## 2         893      3             Wilkes, Mrs. James (Ellen Needs) female
    ## 3         894      2                    Myles, Mr. Thomas Francis   male
    ## 4         895      3                             Wirz, Mr. Albert   male
    ## 5         896      3 Hirvonen, Mrs. Alexander (Helga E Lindqvist) female
    ## 6         897      3                   Svensson, Mr. Johan Cervin   male
    ##    Age SibSp Parch  Ticket    Fare Cabin Embarked Survived
    ## 1 34.5     0     0  330911  7.8292              Q       NA
    ## 2 47.0     1     0  363272  7.0000              S       NA
    ## 3 62.0     0     0  240276  9.6875              Q       NA
    ## 4 27.0     0     0  315154  8.6625              S       NA
    ## 5 22.0     1     1 3101298 12.2875              S       NA
    ## 6 14.0     0     0    7538  9.2250              S       NA

#### Combining the 2 datasets for detailed analysis.

To distinguish the rows after combining to which dataset(train/test) they belong, we create another column which stores information about "belongs to train set" or "belongs to test set".

``` r
train_data$belongs_to_train <- TRUE
tail(train_data)
```

    ##     PassengerId Survived Pclass                                     Name
    ## 886         886        0      3     Rice, Mrs. William (Margaret Norton)
    ## 887         887        0      2                    Montvila, Rev. Juozas
    ## 888         888        1      1             Graham, Miss. Margaret Edith
    ## 889         889        0      3 Johnston, Miss. Catherine Helen "Carrie"
    ## 890         890        1      1                    Behr, Mr. Karl Howell
    ## 891         891        0      3                      Dooley, Mr. Patrick
    ##        Sex Age SibSp Parch     Ticket   Fare Cabin Embarked
    ## 886 female  39     0     5     382652 29.125              Q
    ## 887   male  27     0     0     211536 13.000              S
    ## 888 female  19     0     0     112053 30.000   B42        S
    ## 889 female  NA     1     2 W./C. 6607 23.450              S
    ## 890   male  26     0     0     111369 30.000  C148        C
    ## 891   male  32     0     0     370376  7.750              Q
    ##     belongs_to_train
    ## 886             TRUE
    ## 887             TRUE
    ## 888             TRUE
    ## 889             TRUE
    ## 890             TRUE
    ## 891             TRUE

``` r
test_data$belongs_to_train <- FALSE
tail(test_data)
```

    ##     PassengerId Pclass                           Name    Sex  Age SibSp
    ## 413        1304      3 Henriksson, Miss. Jenny Lovisa female 28.0     0
    ## 414        1305      3             Spector, Mr. Woolf   male   NA     0
    ## 415        1306      1   Oliva y Ocana, Dona. Fermina female 39.0     0
    ## 416        1307      3   Saether, Mr. Simon Sivertsen   male 38.5     0
    ## 417        1308      3            Ware, Mr. Frederick   male   NA     0
    ## 418        1309      3       Peter, Master. Michael J   male   NA     1
    ##     Parch             Ticket     Fare Cabin Embarked Survived
    ## 413     0             347086   7.7750              S       NA
    ## 414     0          A.5. 3236   8.0500              S       NA
    ## 415     0           PC 17758 108.9000  C105        C       NA
    ## 416     0 SOTON/O.Q. 3101262   7.2500              S       NA
    ## 417     0             359309   8.0500              S       NA
    ## 418     1               2668  22.3583              C       NA
    ##     belongs_to_train
    ## 413            FALSE
    ## 414            FALSE
    ## 415            FALSE
    ## 416            FALSE
    ## 417            FALSE
    ## 418            FALSE

#### Perform vertical join on 2 datasets to combine and summarize data. And check head/tail or counts of belongs\_to\_train column to verify if the datasets were combined properly.

``` r
TitanicFull <- rbind(train_data,test_data)
head(TitanicFull)
```

    ##   PassengerId Survived Pclass
    ## 1           1        0      3
    ## 2           2        1      1
    ## 3           3        1      3
    ## 4           4        1      1
    ## 5           5        0      3
    ## 6           6        0      3
    ##                                                  Name    Sex Age SibSp
    ## 1                             Braund, Mr. Owen Harris   male  22     1
    ## 2 Cumings, Mrs. John Bradley (Florence Briggs Thayer) female  38     1
    ## 3                              Heikkinen, Miss. Laina female  26     0
    ## 4        Futrelle, Mrs. Jacques Heath (Lily May Peel) female  35     1
    ## 5                            Allen, Mr. William Henry   male  35     0
    ## 6                                    Moran, Mr. James   male  NA     0
    ##   Parch           Ticket    Fare Cabin Embarked belongs_to_train
    ## 1     0        A/5 21171  7.2500              S             TRUE
    ## 2     0         PC 17599 71.2833   C85        C             TRUE
    ## 3     0 STON/O2. 3101282  7.9250              S             TRUE
    ## 4     0           113803 53.1000  C123        S             TRUE
    ## 5     0           373450  8.0500              S             TRUE
    ## 6     0           330877  8.4583              Q             TRUE

``` r
tail(TitanicFull)
```

    ##      PassengerId Survived Pclass                           Name    Sex
    ## 1304        1304       NA      3 Henriksson, Miss. Jenny Lovisa female
    ## 1305        1305       NA      3             Spector, Mr. Woolf   male
    ## 1306        1306       NA      1   Oliva y Ocana, Dona. Fermina female
    ## 1307        1307       NA      3   Saether, Mr. Simon Sivertsen   male
    ## 1308        1308       NA      3            Ware, Mr. Frederick   male
    ## 1309        1309       NA      3       Peter, Master. Michael J   male
    ##       Age SibSp Parch             Ticket     Fare Cabin Embarked
    ## 1304 28.0     0     0             347086   7.7750              S
    ## 1305   NA     0     0          A.5. 3236   8.0500              S
    ## 1306 39.0     0     0           PC 17758 108.9000  C105        C
    ## 1307 38.5     0     0 SOTON/O.Q. 3101262   7.2500              S
    ## 1308   NA     0     0             359309   8.0500              S
    ## 1309   NA     1     1               2668  22.3583              C
    ##      belongs_to_train
    ## 1304            FALSE
    ## 1305            FALSE
    ## 1306            FALSE
    ## 1307            FALSE
    ## 1308            FALSE
    ## 1309            FALSE

``` r
table(TitanicFull$belongs_to_train)
```

    ## 
    ## FALSE  TRUE 
    ##   418   891

#### Dealing with missing values

``` r
table(TitanicFull$Embarked)
```

    ## 
    ##       C   Q   S 
    ##   2 270 123 914

2 observations belong to a missing category. Let us try to add those two observations to "S" or most frequent category.

``` r
TitanicFull[TitanicFull$Embarked == '', "Embarked"] <- 'S'
table(TitanicFull$Embarked)
```

    ## 
    ##   C   Q   S 
    ## 270 123 916

Age has missing values too. Around 263 are NA. Find out first the median of the full dataset and replace NA with that median.

``` r
table(is.na(TitanicFull$Age))
```

    ## 
    ## FALSE  TRUE 
    ##  1046   263

``` r
median(TitanicFull$Age, na.rm  = TRUE)
```

    ## [1] 28

``` r
TitanicFull[is.na(TitanicFull$Age), "Age"] <- median(TitanicFull$Age, na.rm  = TRUE)
table(is.na(TitanicFull$Age))
```

    ## 
    ## FALSE 
    ##  1309

Fare has missing values too.Find out first the median of the full dataset and replace NA with that median.

``` r
table(is.na(TitanicFull$Fare))
```

    ## 
    ## FALSE  TRUE 
    ##  1308     1

``` r
median(TitanicFull$Fare, na.rm  = TRUE)
```

    ## [1] 14.4542

``` r
TitanicFull[is.na(TitanicFull$Fare), "Fare"] <- median(TitanicFull$Fare, na.rm  = TRUE)
table(is.na(TitanicFull$Fare))
```

    ## 
    ## FALSE 
    ##  1309

Now, we can convert some of the variables to factors. We can do this using as.factors function.

``` r
TitanicFull$Pclass <- as.factor(TitanicFull$Pclass)
TitanicFull$Pclass <- as.ordered(TitanicFull$Pclass)
TitanicFull$Sex <- as.factor(TitanicFull$Sex)
TitanicFull$Embarked <- as.factor(TitanicFull$Embarked)
```

Dataset is now ready for building predictive models. Prior to that, we need to separate the dataset into train and test again. Let's leverage the "belongs\_to\_train" column to segregating train and test data.

``` r
train <- TitanicFull[TitanicFull$belongs_to_train==TRUE,]
summary(train)
```

    ##   PassengerId       Survived      Pclass      Name               Sex     
    ##  Min.   :  1.0   Min.   :0.0000   1:216   Length:891         female:314  
    ##  1st Qu.:223.5   1st Qu.:0.0000   2:184   Class :character   male  :577  
    ##  Median :446.0   Median :0.0000   3:491   Mode  :character               
    ##  Mean   :446.0   Mean   :0.3838                                          
    ##  3rd Qu.:668.5   3rd Qu.:1.0000                                          
    ##  Max.   :891.0   Max.   :1.0000                                          
    ##       Age            SibSp           Parch           Ticket         
    ##  Min.   : 0.42   Min.   :0.000   Min.   :0.0000   Length:891        
    ##  1st Qu.:22.00   1st Qu.:0.000   1st Qu.:0.0000   Class :character  
    ##  Median :28.00   Median :0.000   Median :0.0000   Mode  :character  
    ##  Mean   :29.36   Mean   :0.523   Mean   :0.3816                     
    ##  3rd Qu.:35.00   3rd Qu.:1.000   3rd Qu.:0.0000                     
    ##  Max.   :80.00   Max.   :8.000   Max.   :6.0000                     
    ##       Fare           Cabin           Embarked belongs_to_train
    ##  Min.   :  0.00   Length:891         C:168    Mode:logical    
    ##  1st Qu.:  7.91   Class :character   Q: 77    TRUE:891        
    ##  Median : 14.45   Mode  :character   S:646    NA's:0          
    ##  Mean   : 32.20                                               
    ##  3rd Qu.: 31.00                                               
    ##  Max.   :512.33

``` r
nrow(train)
```

    ## [1] 891

``` r
test <- TitanicFull[TitanicFull$belongs_to_train==FALSE,]
summary(test)
```

    ##   PassengerId        Survived   Pclass      Name               Sex     
    ##  Min.   : 892.0   Min.   : NA   1:107   Length:418         female:152  
    ##  1st Qu.: 996.2   1st Qu.: NA   2: 93   Class :character   male  :266  
    ##  Median :1100.5   Median : NA   3:218   Mode  :character               
    ##  Mean   :1100.5   Mean   :NaN                                          
    ##  3rd Qu.:1204.8   3rd Qu.: NA                                          
    ##  Max.   :1309.0   Max.   : NA                                          
    ##                   NA's   :418                                          
    ##       Age            SibSp            Parch           Ticket         
    ##  Min.   : 0.17   Min.   :0.0000   Min.   :0.0000   Length:418        
    ##  1st Qu.:23.00   1st Qu.:0.0000   1st Qu.:0.0000   Class :character  
    ##  Median :28.00   Median :0.0000   Median :0.0000   Mode  :character  
    ##  Mean   :29.81   Mean   :0.4474   Mean   :0.3923                     
    ##  3rd Qu.:35.75   3rd Qu.:1.0000   3rd Qu.:0.0000                     
    ##  Max.   :76.00   Max.   :8.0000   Max.   :9.0000                     
    ##                                                                      
    ##       Fare            Cabin           Embarked belongs_to_train
    ##  Min.   :  0.000   Length:418         C:102    Mode :logical   
    ##  1st Qu.:  7.896   Class :character   Q: 46    FALSE:418       
    ##  Median : 14.454   Mode  :character   S:270    NA's :0         
    ##  Mean   : 35.577                                               
    ##  3rd Qu.: 31.472                                               
    ##  Max.   :512.329                                               
    ## 

``` r
nrow(test)
```

    ## [1] 418

"Survived" needs to be converted to categorical variable

``` r
train$Survived <- as.factor(train$Survived)
str(train$Survived)
```

    ##  Factor w/ 2 levels "0","1": 1 2 2 2 1 1 1 1 2 2 ...

#### Random Forest Modeling

``` r
formula1 <- as.formula("Survived ~ Pclass + Sex + Age + SibSp + Parch + Embarked")
# Using Random Forest
require(randomForest)
```

    ## Loading required package: randomForest

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
TitanicModelRF <- randomForest(formula = formula1, data = train, ntree = 400, ntry = 3, nodesize = 0.01*nrow(test))
FeaturesEq <- "Pclass + Sex + Age + SibSp + Parch + Embarked"
Survived <- predict(TitanicModelRF, newdata = test)

# Creating output dataset
PassengerId <- test$PassengerId
OutputDF <- as.data.frame(PassengerId)
OutputDF$Survived <- Survived
head(OutputDF)
```

    ##   PassengerId Survived
    ## 1         892        0
    ## 2         893        0
    ## 3         894        0
    ## 4         895        0
    ## 5         896        0
    ## 6         897        0

``` r
tail(OutputDF)
```

    ##     PassengerId Survived
    ## 413        1304        0
    ## 414        1305        0
    ## 415        1306        1
    ## 416        1307        0
    ## 417        1308        0
    ## 418        1309        1

Write kaggle submission csv file

``` r
write.csv(OutputDF, file = "kaggle_submission.csv", row.names = FALSE)
```
