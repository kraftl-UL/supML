﻿#  
# Supervised Machine Learning
**University of Lucerne | HS21** 
**by Dr. Massimo Mannino**

### <u> A project by Dominik Walter, Frederik Blümel & Ludwig Kraft


## Overview
1. Introduction and Motivation
2. Data Cleaning
3. Modeling
4. Comparison and Conclusion


## 1 Introduction and Motivation

In the course Supervised Machine Learning we (the authors) have gained valueable insights into various spheres of supervised machine learning (ML). 

In greater detail we first dove into linear and logistic regression models, learned to assess model accuracy and evaluate models based on metrics like Confusion Matrix and its implicated metrics like Precision, Recall and the  ROC curve. We later looked into sampling strategies like cross validation and bootstrapping to handle large datasets more efficiently. Model selection procedures like lasso and ridge regression led us further to the decision tree techniques of random forests.

Accompanying the theoretical set-up of the course we enjoyed (and endured) a practical application of those supervised ML models. 

The project aims to train a decent model and predict rent prices of Switzerland's housing market based on a variety of variables.

To derive at a decent model we compare at least four different ones and apply cross validation for resampling from our data for at least two models. 

In this paper we follow along our path of code and display central thoughts of why we proceded as we did. This includes some descriptive statistics, graph interpretations and derived measures of aforementioned. 
Additionally, we briefly explain the techniques we utilized to underline our comprehensive understanding of the models' conceptual baselines. 
Finally, we assess our results and apply our best model to a new dataset to predict the outcome. 

The dataset in question contains 90001 real-life observations of Switzerland's housing market described and structured into 100 different variables. 


## 2 Data Cleaning

The dataset is not clean in its original form and requires a decent amount of cleaning. 

We want to run our model on a holistic set of data and therefore aim towards a reduced dataset with relevant variables, predominantly meaningful observations and entries in every cell.

Obviously, we cannot have each and every discussion on which variables to delete and which not due to the given limited extent of this paper. However, we attempt to be as detailed as possible. 

### War on redundant variables 

We need to understand that many of the collected observations have not been filled entirely by the creators of the observations. 
Does an NA in *bath* entail that the flat has no bathroom? There rather is a bathroom, but maybe not three as three bathrooms would have been valuable to mention when uploading housing file.

There exist some extreme cases as the variables *public_transport*, *wheelchair* or *garden_m2* with no entry at all. We delete those variables just like the ones containing less than 500 entries as we do not assess them to add any value to the model. Those 10 include heating options, *middle_house*, *sunny*, *furnished*, *bath* or *bright*. 

Another 34 variables have less than 5000 entries. We delete them as well. They include *shower*, *middle_house*, *oven*, etc.
We assume many of those variables not to be properly filled by the observations' creators. Otherwise a majority of Swiss houses and flats would not have, e.g. a shower.

We also delete the subcategories of *Mirco_rating*, but leave itself. This leaves us with 51 variables. 

###  Topic of Multicollinearity 

Two of the variables being highly correlated induces the possibiliy of multicollinearity.
We check for collinearity amongst the remainder variables. 

![enter image description here](https://github.com/kraftl-UL/supML/blob/main/images/corrplot_v1.png?raw=true)

We note several highly correlated variables as a supermarket and restaurant per km2 counter or the month and quarter of the time the observation was created. There even exist variables holding the same information twice as the *sonnenklasse* descriptor. We delete amongst the aforementioned or *avg_bauperiode* and *dist_to_school_1* 20 more variables. Which leaves us with 31. 

Multicollinearity appears not that intensively anymore as before. Strong correlation remains with variables that are obvious as with *rooms* and *area* or variables that will be omitted in the final model as *GDENR* and *msregion*, which are representatives of the observations's location. We also note the high correlation between *area* and the dependent variable *rent_full*.

![Collinearity after deleting redundant variables](https://github.com/kraftl-UL/supML/blob/main/images/corrplot_v2.png?raw=true)

Testing statistically on Multicollinearity we utilize the vif-test test. Variance inflation factor. With a simple linear model we check the 1/vif(model) values not be less than 0.1. This holds. 


Ultimately the following variables remain:

*GDENAMK*, *GDENR*,  *KTKZ*, *area*  , *balcony*,  *elevator*,*floors*,  *home_type*,  *msregion*, *parking_indoor* , *parking_outside*,  *rent_full*, *rooms* , *year_built*,  *Micro_rating*, *wgh_avg_sonnenklasse_per_egid*, *Anteil_auslaend*,  *Avg_age*, *Avg_size_household*, *Noise_max*,  *anteil_efh*, *avg_anzhl_geschosse*, *dist_to_4G*,  *dist_to_5G*, *dist_to_haltst*  , *dist_to_highway* , *dist_to_lake*, *dist_to_main_stat*,  *dist_to_train_stat* , *superm_pix_count_km2*, *dist_to_river*  and *na_count*.  


### War on NAs 

However, with those 31 variables we still have 18 with NAs. 

Then, we tailor the way to substitute the missing values for each variable - mostly based on an idea of the variable's distribution (histograms, boxplots, unique values, common sense).

Tackling sparse observations, we delete these rows as we would create synthetic rows by imputing. Also, we delete rows which contain NAs in area and rooms, since we seek to keep the risk low to be biased, especially in the regarding the variables of rooms and area.

For discrete values as *balcony*, *elevator*, *parking_indoor*, *parking_outside* we consider an NA to be 0 as it is likely to not type anything when creating an observation, than typing 0 for non-existent paramters.
For missing *floor* values we considered the median. 

From external research we know that there doesn't exist any location in Switzerland with a distance of more than 16km to the next lake. Everything above this distance in our dataset - concerning *dist_to_lake* - is set to NA and later all NAs are replaced by the median, which is ~1206m.

We also assume the average age of people living in a hectare around a certain ovservation is very unlikely to be lower than 18 as before adulthood people cannot rent an appartment or flat. Even if we take the chance of having babies into account, the likelihood of all people in an area of one hectar being younger than 18 is very low. The average of those people in a certain area will be even higher. Same applies for the upper egde of *Avg_age*. So we first reduce the top and bottom outliers by more appropriate values of 25 & 58 and fill the remaining NAs with the (more representative) median value, which is 41.84 years.

Observations with *area* values lower than 6.0m2 were handled as outliers and deleted.

As we deleted the few observations with NAs in both, *area* and *rooms*, we refill *area*  depending on the number of rooms and a average room size computed based on observations with both entries. Vice versa for NAs in *rooms*. 

In *rooms* there were many obviously wrong entries (e.g. 1.6, 2.4, 3.1 or 5.4). We rounded those to the nearest .5 value. 

This was a short overview of methods and cherry-picking of variables how we handled NA substitution. 

We end the cleaning process with a dataset of 88921 observations and 31 variables. 

### Descriptive statistics 

In the following we will take a more detailed look into the variables we are using for our predictive models. During the process of data cleaning, we already took a first look into the data and focused on outliers of the variables. 

Now, we aim to understand the data in greater detail. Therefore, we will investigate some variables representatively, and share the insights that appear to be valuable.

![enter image description here](https://github.com/kraftl-UL/supML/blob/main/images/KTKZ~count.png?raw=true)

Looking at the distribution of cantons within the dataset shows that by far most of the apartments are advertised in the canton of Zurich. Bern and Vaud have many observations, too. These three cantons together make up a larger portion than 40% of all advertisements. Together with the next canton, Aargau, these apartments make 50% of all advertisements. 

In combination with a boxplot over the rent prices per canton, this gives further insight into the distribution of rent prices within the data.

![enter image description here](https://github.com/kraftl-UL/supML/blob/main/images/Box_KTKZ~rent_full.png?raw=true)

We can see that the cantons of Geneva, Zug and Zurich tend to be the most expensive cantons in Switzerland. Additionally, we note that the average rent prices per canton lie rather close together.

As we saw in the correlation plot from above, area and room seem to have a strong impact on the rent price. Therefore, we plot both these variables pairwise.

![enter image description here](https://github.com/kraftl-UL/supML/blob/main/images/area~rent_full.png?raw=true)

Even though we note heteroscedasticity, rent price and the area of an apartment are positively correlated. It seems like when reaching a certain point which is around 200m2, the impact of area on the price remains positive but decreases. This could mean that bigger apartments are more available in areas where rent prices are lower. Given the graph from above, these might be more rural than urban areas.

 The relation between rooms and the rent prices is similar to the one from above.
 
 Common sense explains that apartments with more rooms tend to be bigger and tend to be more expensive. Interestingly, as with the area of the apartment, the impact of the number of rooms on the rent price tends to decrease from a certain number of rooms.

![enter image description here](https://github.com/kraftl-UL/supML/blob/main/images/rooms~rent_full.png?raw=true)

Next, we look at the relation of Micro_rating and the rent price. In the first place, we assumed the Micro_rating has an impact on the rent price. 
The low number of NAs and the fact that we already deal with a rating element here, lead to this assumption. 

However, the graph points to a different situation. We cannot note a clear correlation here but rather observe heteroscedasticity here, too.

![enter image description here](https://github.com/kraftl-UL/supML/blob/main/images/Micro_rating~rent_full.png?raw=true)


After having thoroughly examined the data set, we got a good impression of what the dataset is all about. Now we are prepared to continue with training of our models.



## 3 Modeling 

After having analyzed the data, we then continue with the actual modeling. 
We tested a multi-linear model, lasso and ridge regression, bagging and a random forest model.

### Multi-linear Model 

We fitted a linear model all variables (except the *GDENR* and *KTKZ* as they would entail too many additional regressors, exceeding our computational power) with the $lm()$ method. This leaves us with 27 independent variables to determine $Y = rent\_ full$. The model we're fitting therefore is of the form

$$
\ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + + \beta_{27} X_{27} + \epsilon
$$

 $X_i$ are vectors of length $88'855$. Some $X_i$ for $i \in \{floors, home\_type, msregion, avg\_anzahl\_geschosse\}$ are discrete regressors due to their factor nature. Some of those turn out to be not significant ( e.g. *avg_anzahl_geschosse*(4,5,6), *msregion*68, *floors*(1,2,3,10,11)). 

We result in a model with 175 regressors of which only 32 are not significant on a 1% level. 



Surprisingly, *balcony* is not significant. We remain with the model statistics for later comparison Residual Mean Squared Error and the R$^2$ for explainability of the models variance. 


For better model fitting we utilized 5-fold cross validation as a sampling strategy. In short, this means, that we devided the data set into 5 equally sized parts, trained a model with the first four and evaluated it on the fifth. 
This process is repeated 5 times, such that each set was the test set exactly once. An average error term is calculated for each training and testing lap.
Ultimately, the RMSE of the model is calculated as the average error of the five validation processes from cross validation.

More information on the resulting model can be checked by running the code. 


|                |RMSE                         |R$^2$|
|----------------|-------------------------------|-----------------------------|
|Multi-linear Model|`378.09`            |`0.68`            |



We attempted a similar model with forward selection as well. 
With forward selection, we attempt to choose significant variables for the model. We fit a model with a first variable, the one that returns the lowest Mean Squared Error.
Then a second variable is added, again selecting the model that gives again the lowest RMSE when controlling for both variables in combination. 
Due to this variable selection method firward selection is also called to be "greedy".
We received worse results.

|                |RMSE                         |R$^2$|
|----------------|-------------------------------|-----------------------------|
|Multi-linear Model|`471.28`            |`0.51`            |




### Polynomial Regression Models

To get a better feeling of how the variables are influencing the dependent variable, we also played around with polynomial regressions. Based on naive estimates of scatter plots of variables on $rent\_full$ we added quadratic terms to a regression model and compared RSMEs. 

We fitted three models:
First, we choose only variables with rather obvious linear relation to rent_full. All other varibales were accounted for in quadratic terms. 
The variables of interest were $dist\_to\_lake$, $Noise\_max$, $Anteil\_auslaend$, $Avg\_age$, $anteil\_efh$, $Micro\_rating$.
The model performed worse than the normal linear one from above. 

With the second model we planned to determine the influence of all variables we used quardatic before to determine the significance of variables with no linear connection. 
We found an  infinitesimally small $R^2$ value of only 1%

In the third model we fitted all in quadratic form. The performance was worse than before as well.


|                |RMSE                         |R$^2$|
|----------------|-------------------------------|-----------------------------|
|Multi-polynomial Model1|`451.67`            |`0.55`            |
|Multi-polynomial Model2|`666.92`            |`0.01`            
|Multi-polynomial Model3|`495.78`            |`0.45`            |



### Lasso Regression 
Next we focused on the Lasso regression. Abbreviated for Least Absolute Shrinkage and Selection Operator, Lasso serves as a model selection device to enhance the prediction accuracy and interpretability of a model. Operationally, it minimizes the sum of squared residuals, subject to a penalty on the absolute size of the regression coefficients, which is prosa for 

$$
min_{\beta}\{ \sum_{i=1}^N(Y_i-\gamma-\sum_{j=1}^pX_{ij}\beta_j)^2 + \lambda \sum_{j=1}^p|\beta_j| \} 
$$ for $\gamma = 0.1/log(N)$ and $N$ the number of observations. The tuning parameter $\lambda$ controls the relative impact of penalty term on the regression coefficient estimates. We obtain it by cross validation. In our very test run, we obtained a $\lambda_{min} =0.0539$.

We split our data with a 80:20 ratio into training and test set.
For the $\lambda_{min} we only get rid of 2 variables for our $\lambda_{min}$: floor26 and floor18. 

In case we would decide for a larger weight for the penalty term, we would lose more variables, which would correspond to shifting further to the right within the following graph. The number of lines in a certain cross section at a certain $\lambda$ represents the number of variables still in the model.

Plotting the Lasso regression:

![enter image description here](https://github.com/kraftl-UL/supML/blob/main/images/lasso_shrinkage.png?raw=true)

The colored lines represent the values taken by a different coefficients in the model. $\lambda$ in the x-axis as the weight for the regularization term. As lambda approaches zero, the loss function of the model approaches the standard  OLS model from above. With lambda increasing we set more and more terms to (almost) zero. 


However, we receive prediction statistics of 
|                |RMSE                         |R$^2$|
|----------------|-------------------------------|-----------------------------|
|Lasso|`378.12`            |`0.68`           |


### Ridge Regression 

Without loosing too many variables with Lasso, we can apply a ridge regression as well. The only difference between both models is the penalty term. It minimizes the sum of squared residuals, subject to a penalty term - not on the absolute size - but on the squared values of the regression coefficients: 

$$
min_{\beta}\{ \sum_{i=1}^N(Y_i-\gamma-\sum_{j=1}^pX_{ij}\beta_j)^2 + \lambda \sum_{j=1}^p\beta_j^2 \} 
$$ for $\gamma = 0.1/log(N)$ and $N$ the number of observations as above. 

Again, we split our data with a 80:20 ratio into training and test set. Here, the tuning parameter $\lambda$ has a larger weight than for the lasso penalty term. Once more, we obtain it by cross validation. $\lambda_{min} =44.7354$.

With *ridge regression* we lose the variable floors26.

Plotting the ridge regression, we note that values are approaching more assymptotically than with lasso which corresponds to the algorithm's theory. 


![enter image description here](https://github.com/kraftl-UL/supML/blob/main/images/ridge_shrinkage.png?raw=true)

When predicting on the test set we have slightly worse performance measures as with lasso:

|                |RMSE                         |R$^2$|
|----------------|-------------------------------|-----------------------------|
|Ridge|`385.19`            |`0.67`            |


### Random Forest 


One of the most widely used and powerful machine learning approaches is the Random Forest algorithm. 
It's a sort of bagging that is used with decision trees. 
It entails continually using various bootstrapped subsets of the data and averaging the models to create multiple separate decision tree models from a single training data set. 
Each tree is constructed independently of the others. 

Random forests can be used for classification tasks (predicting a categorical variable), but regression models (predicting a continuous variable), too.
The difference between Bagging and RandomForest is mainly that in Bagging all of the predictors are considered for each node for a split. 
With Bagging having all predictors used at all times it tends to overfitting. Therefore, the function graph can "adapt" better or more 
accurately to the observations, which should lead to an overfit at a certain point. 

In a first run, we applied the code from the RandomForest package to bypass the BlackBox caret for once. 
Later, however, we reverted to caret, since the optimal value for mtry can be calculated more easily with this package. 
 
In general, the main thing you need to do to avoid overfitting in a random forest is optimize a tuning parameter (mtry) 
that governs the number of features that are randomly chosen to develop each tree from the bootstrapped data. 
Typically, this is done using k-fold cross-validation, where k lays between 5 and 10. 
Furthermore, increasing the size of the forest improves forecast accuracy. 
However, there are normally diminishing benefits after you reach a few hundreds of trees.

For fine-tuning of ntrees we have been looping over the parameter. We figured that when considering more than 140 trees, we only achieved marginal performance improvements.
This is also displayed here.

![enter image description here](https://github.com/kraftl-UL/supML/blob/main/images/rf_performance_improv.jpeg?raw=true)

Now display the model statistics for a short selection of tuning-parameters.



|                |RMSE                         |R$^2$|
|----------------|-------------------------------|-----------------------------|
|Random Forest1 (max_nodes = 5000, ntrees = 10)|`346.88`            |` 0.69`             |
|Random Forest2 (max_nodes = 5000, ntrees = 140)|`335.97`            |` 0.75`             |
|Random Forest3 (max_nodes = 5000, ntrees = 200)|`336.36`            |` 0.76`             |
|Random Forest4 (max_nodes = 15000, ntrees = 140)|`330.00`            |` 0.76`             |
|Random Forest5 (max_nodes = 5000, ntrees = 140) (bagging: mtry = 27)|`339.74`            |` 0.75`             |


Random Forest(3,4,5) took extensive computational time due to the rise in max_nodes or ntrees. We made those calculations to figure whether an increase of nods would improve the model substantially. This was not the case.
Therefore we conclude the best random forest due to the best mix of RMSE, R$^2$ and computational time to be Random Forest2.


## Comparison and Conclusion


After having trained and evaluated all models based on the size of error terms, we decide to make the prediction by using Random Forest2.


|                |RMSE                         |R$^2$|
|----------------|-------------------------------|-----------------------------|
|Multi-linear Model|`378.09`            |`0.68`            |
|Multi-polynomial Model1|`451.67`            |` 0.55`             |
|Lasso|`378.12`            |` 0.68`             |
|Ridge|`385.19`            |` 0.67`             |
|Random Forest2 (max_nodes = 5000, ntrees = 140)|`335.97`            |` 0.75`             |


With the random forest we receive not only the lowest Residual Mean Squared Error, but the highest explainability of the prediction's variance, too. 

Therefore, we use the random forest to predict the original holdout set X_test.
The predicted values can be found in the corresponding [repository](https://github.com/kraftl-UL/supML). 

This concludes our project and paper. We thank all readers kept on reading until down here and wish a good remaining semester. 


