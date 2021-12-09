library(tidyverse) 
library(psych) 
library(corrplot) 
library(margins)
library(ggplot2)
library(Hmisc)
library(ggcorrplot)
library(hot.deck)
library(dplyr)
library(plyr)
library(car)
library(dlookr)
library(readxl)
library(qpcR)
library(caret)
library(glmnet)
library(klaR)
library(randomForest)
library(ROCR)
library(ranger)
library(reshape)
library(boot)
library(lmvar)
library(corpcor)
library(mctest)
library(boot)
library(randomForestExplainer)

#install.packages("addYourPackage")


getwd()
setwd("C:/Users/kraft/OneDrive/Desktop/Bewerbung/UL/3_HS_2021/Supervised Learning/Final Project")
list.files()
list.files(pattern=".csv")



data_basis <- as.data.frame(read.csv("training.csv"))
df <- data_basis

#reorder
df = df %>% relocate("rent_full", .after = last_col())

dim(df)
str(df)
#View(df)


# We consider the extent of the dataset as to be too vast. 
# Therefore, we use various approaches to reduce first, the number of observations and, 
# second, the number of variables.


# Counting for features with extreme high number of NAs. These will later be
# excluded, since we can assume missing data in some cases rather than scarce features
na_count <- sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)
na_count_start = na_count


# NA count gives 10 features with no data at all. These will be deleted first
length(which(na_count < 90001))

df <- subset(df, select = -c(which(na_count == 90001)))

#update na_count
na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)


# Next, we delete all features with less than 5000 entries, after checking each
# of the features individually
df <- subset(df, select = -c(which(na_count > 85001)))

#update na_count
na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)




# Since micro_rating is the weighted average of the features Micro_rating_NoiseAndEmission,
# Micro_rating_Accessibility, Micro_rating_DistrictAndArea, Micro_rating_SunAndView
# and Micro_rating_ServicesAndNature, we will delete those

df <- subset(df, select = -c(Micro_rating_NoiseAndEmission,
                             Micro_rating_Accessibility, 
                             Micro_rating_DistrictAndArea, 
                             Micro_rating_SunAndView, 
                             Micro_rating_ServicesAndNature))
#update na_count
na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)


# To get rid of unneccessary variables we now look out for strong correlation with a large corrplot.
cor(df[, unlist(lapply(df, is.numeric))], use = "pairwise.complete.obs")
ggcorrplot(cor(df[, unlist(lapply(df, is.numeric))], use = "pairwise.complete.obs"), method = "circle")



# Delete further features that won't add value to the model
df <- subset(df, select = -c(descr, newly_built, year, quarter_specific, address,
                             date, lat, lon, area_useable, cabletv, month, quarter_general, 
                             quarter_specific, year, apoth_pix_count_km2, 
                             restaur_pix_count_km2, avg_bauperiode, dist_to_school_1, 
                             geb_wohnnutz_total, wgh_avg_sonnenklasse_per_egid.1, kids_friendly, 
                             max_guar_down_speed))

na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)

# overview relevant variables
colnames(df)


dim(df)
str(df)
#View(df)


# deleting of columns done
##########################################################



# correlation
#############


# Take another glance at correlations between variables
# this plot has been used before for determining variables to be deleted due to multicorrelations
cor(df[, unlist(lapply(df, is.numeric))], use = "pairwise.complete.obs")
ggcorrplot(cor(df[, unlist(lapply(df, is.numeric))], use = "pairwise.complete.obs"), method = "circle")

# The corrplot gives a hint of which variables have an explanatory value for the 
# target variable rent_full. (Those correlating with rent_full). 
# Area and rooms seem to be promising by checking by looking at a first glance.

# Now we count the NAs in each row. If we detect rows with multiple NAs, we can
# delete these rows as we would create synthetic rows by imputing. Also, we delete
# rows which contain NAs in area and rooms, since we seek to keep the risk low to
# be biased, especially in the regarding the variables  of rooms and area.

df$na_count <- apply(df, 1, function(x) sum(is.na(x)))

# We now delete rows with NAs in area and rooms

df = subset(df, area != "NA" | rooms != "NA")

# After counting the NAs per row, we  decide not to delete further rows, since the
# maximum number of NAs is 15 and we assume, this is a proper number to be replaced
# with values rather than deleted. Also, we ensured before, that no row has neither
# a value in rooms nor in area.



# more data cleaning
####################
####################



#substitute NAs
################
attach(df)


#discrete variables NAs to 0
# assumption: where no discrete variable is displayed, we assume it to be non-existent. 
discrete_vars = c("balcony", "elevator", "parking_indoor", "parking_outside")
df[discrete_vars][is.na(df[discrete_vars])] = 0



# floors 
# substitute NAs with median of all floors. 
summary(df$floors)
df$floors[is.na(df$floors)] = median(df$floors, na.rm = TRUE)


# year_built
# there are outliers and many NAs in year_built. We first remove outliers and
# the substitute NAs with the most frequent year_built value, which is

summary(df$year_built)
boxplot(df$year_built)
count(df$year_built < 1600) # delete 69 observations as outliers
df = subset(df, year_built >= 1600 | is.na(year_built))

summary(df$year_built)
hist(df$year_built)
# We do not want to use the median, as it is still skewed. Therefore we decide to plot the 10 most frequent 
# entries and and take a weighted mean from those. This is 2016 (rounded). 

sort(table(df$year_built),decreasing=TRUE)[1:10]
weighted.mean(c(2019, 2018, 2016, 2017, 2015, 2014, 2013, 2012, 2011, 2010), 
              c(2289, 1370, 1258, 1247, 1085, 1009,  885,  772,  740,  588 ))
# == 2016

# Then we substitute all NAs by 2016.

df$year_built[is.na(df$year_built)] = 2016


# dist_to_lake
# after online research we know that no village is more than 16km from a lake. 
# we delete all outliers > 16'000

df = subset(df, dist_to_lake <= 16000 | is.na(dist_to_lake))
summary(dist_to_lake)

# Then we substitute all NAs by the median.
df$dist_to_lake[is.na(df$dist_to_lake)] = median(df$dist_to_lake, na.rm = TRUE)
summary(dist_to_lake)




# dist_to_main_stat
summary(dist_to_main_stat)
boxplot(dist_to_main_stat)
hist(dist_to_main_stat)

# values seem to be appropriate.  Now NA handling.
# We substitute NAs with the median because the mean is skewed due to the relatively large values.

df$dist_to_main_stat[is.na(df$dist_to_main_stat)] = median(df$dist_to_main_stat, na.rm = TRUE)


# Avg_size_household
summary(Avg_size_household)
boxplot(Avg_size_household)
hist(Avg_size_household)

# no skewing due to higher values. Therefore we substitute the NAs by the mean.
df$Avg_size_household[is.na(df$Avg_size_household)] = mean(df$Avg_size_household, na.rm = TRUE)


# Anteil_auslaend
summary(Anteil_auslaend)
boxplot(Anteil_auslaend)
hist(Anteil_auslaend)

# we substitute NAs by the mean just like the values with 1.0 as this value is not to be trusted
df$Anteil_auslaend[is.na(df$Anteil_auslaend) | df$Anteil_auslaend == 1] = mean(df$Anteil_auslaend, na.rm = TRUE)


#Avg_age
summary(Avg_age)
boxplot(Avg_age)
hist(Avg_age)

# we decide based on the boxplot and choose avg ages only between 25 
# and 58 and replace them as well as the NAs with the median.
count(Avg_age < 25 | Avg_age > 58)

df$Avg_age[df$Avg_age < 23] = median(df$Avg_age, na.rm = TRUE)
df$Avg_age[df$Avg_age > 58] = median(df$Avg_age, na.rm = TRUE)

#Substitute NAs with mean or median (only 0.01 difference)

df$Avg_age[is.na(df$Avg_age)] = mean(df$Avg_age, na.rm = TRUE)
summary(df$Avg_age)


# avg_anzhl_geschosse

summary(avg_anzhl_geschosse)
boxplot(avg_anzhl_geschosse)
hist(avg_anzhl_geschosse)

# we decide based on the histogram and substitute the NAs with the rounded median.
df$avg_anzhl_geschosse[is.na(df$avg_anzhl_geschosse)] = round(median(df$avg_anzhl_geschosse, na.rm = TRUE))
summary(df$avg_anzhl_geschosse)


# anteil_efh 
summary(anteil_efh)
boxplot(anteil_efh)
hist(anteil_efh)

# we consider the large number of 0 entries (no efh in the hectare) to be representative of the
# housing market in Switzerland and therefore replace all NAs with 0)

df$anteil_efh[is.na(df$anteil_efh)] = 0


# dist_to_haltst
summary(dist_to_haltst)
boxplot(dist_to_haltst)
hist(dist_to_haltst, breaks = 100)

# no outliers as min and max value are reasonable
# we use the median to replace the distance to haltestelle as mean is skewed due to high distance observations
df$dist_to_haltst[is.na(df$dist_to_haltst)] = median(df$dist_to_haltst, na.rm = TRUE)
summary(df$dist_to_haltst)


# wgh_avg_sonnenklasse_per_egid

summary(wgh_avg_sonnenklasse_per_egid)
boxplot(wgh_avg_sonnenklasse_per_egid)
hist(wgh_avg_sonnenklasse_per_egid, breaks = 100)

# we use the mean to replace the NAs. Not really relevant as it's only 21 observations, and the spread is not very large.
df$wgh_avg_sonnenklasse_per_egid[is.na(df$wgh_avg_sonnenklasse_per_egid)] = mean(df$wgh_avg_sonnenklasse_per_egid, na.rm = TRUE)






na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)


# we can now delete the na_count per observation.
df$na_count = NULL


# Now only rooms and area remain as variables with NAs. 

# The easiest way for the detection of multicollinearity is to examine 
# the correlation between each pair of explanatory variables. 
# If two of the variables are highly correlated, then this may the possible 
# source of multicollinearity.


# We consider the vif-test to test for multi-collinearity. variance inflation factor.
# Therefore, we fit an easy linear model.
model_test_mc = lm(rent_full ~ . -GDENAMK-GDENR-KTKZ, df)
summary(model_test_mc)

# No value is supposed to be lower than 0.1. Check. 
1/vif(model_test_mc) 

# No multi-collinearity due to rooms and area. 


# substitute rooms NAs

summary(rooms)
sort(unique(rooms))
# outlier: 15 has a relatively small area. We assume it to be 1.5
df[which(df$rooms == 15),]$rooms = 1.5

# outlier: 11. 130 area and low rent. We delete the observation and get rid of an additional room level.
df = df[-which(df$rooms == 11.0), ]

boxplot(df$rooms)
hist(df$rooms, breaks = 100)


# replace rooms NAs according number of average rooms with
mean_area_per_room = mean(df$area / df$rooms, na.rm = TRUE)
df$rooms[is.na(df$rooms)] = round_any(df$area[is.na(df$rooms)] / mean_area_per_room, 0.5)

# this leaves us with 0.0 and 0.5 rooms, which we now set to 1.0 
df$rooms[df$rooms<1.0] = 1.0


# wrong values: 1.6 2.4, 3.1, 5.4 
# we get rid of those by rounding to the nearest .5
sort(unique(df$rooms))
df$rooms = round_any(df$rooms, 0.5)


# area
summary(area)
boxplot(df$area)
hist(df$area, breaks = 100)


# replace rooms NAs according number of average rooms with
mean_rooms_per_area = mean(df$rooms / df$area, na.rm = TRUE)
df$area[is.na(df$area)] = round_any(df$rooms[is.na(df$area)] / mean_rooms_per_area, 1)


summary(df$area)

# this leaves us still with very small areas like 1, 2, 3 and 5. 
# therefore, we delete those observations from the data. 
df = df[-which(df$area < 6.0), ]


na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)


#deleting of observations done. 



# avg_anzahl_geschosse
# We only accept whole numbers and transform the variable to a factor later. 
df$avg_anzhl_geschosse = round_any(df$avg_anzhl_geschosse, 1)



# Make factors for relevant features that are characters
df$home_type <- as.factor((df$home_type))
df$GDENR = as.factor(df$GDENR)
df$KTKZ = as.factor(df$KTKZ)
df$msregion = as.factor(df$msregion)
df$floors = as.factor(df$floors)
df$avg_anzhl_geschosse = as.factor(df$avg_anzhl_geschosse)


##############################################################
##############################################################



# Descriptive analysis
########################

# Examine histograms for all variables
par(mfrow=c(3,3))
hist.data.frame(df)


# Plot specific relations for further examination
par(mfrow=c(2,1))

# rent~ home_type
plot(df$home_type, df$rent_full, xlab = "msregion", ylab ="rent", varwidth = T, col = c(2:106))

# rent~ msregion
plot(df$msregion, df$rent_full, xlab = "msregion", ylab ="rent", varwidth = T, col = c(2:106))


# in case error under next 4 plots run
# dev.off()


# rent~ area
ggplot(df, aes(x= area, y= rent_full)) + 
  geom_point() +
  geom_smooth()

# rent~ micro_rating
ggplot(df, aes(x= Micro_rating, y= rent_full)) + 
  geom_point() +
  geom_smooth()

#############################################################
##########################################################



# modeling
##########


###########################################################################
####### Multi-linear model with k-fold Cross Validation ##################################################


df_LM_kfold <- subset(df, select = -c(GDENAMK, GDENR, KTKZ))


# Define training control
set.seed(123) 
training.samples <- df_LM_kfold$rent_full %>%
  createDataPartition(p = 0.8, list = FALSE)

train.data  <- df_LM_kfold[training.samples, ]
test.data <- df_LM_kfold[-training.samples, ]

train.control <- trainControl(method = "cv", number = 5)


# Train the model
model <- train(rent_full ~., data = train.data, method = "lm",
               trControl = train.control)

# Summarize the results
lm1 <- as.data.frame(print(model))

summary(model)

predictions <- model %>% predict(test.data)
# Model performance
Reg_stat <- data.frame(
  RMSE = RMSE(predictions, test.data$rent_full),
  R2 = R2(predictions, test.data$rent_full)
)


######################################
## linear model with forward selection

set.seed(123) 
training.samples <- df_LM_kfold$rent_full %>%
  createDataPartition(p = 0.8, list = FALSE)

train.data  <- df_LM_kfold[training.samples, ]
test.data <- df_LM_kfold[-training.samples, ]

train.control <- trainControl(method = "cv", number = 5)

model2 <- train(rent_full~., data=df_LM_kfold, trControl=train.control, method="leapForward")


lm_2 <- as.data.frame(print(model2))

summary(model2)

predictions2 <- model2 %>% predict(test.data)
# Model performance
Reg_stat <- data.frame(
  RMSE = RMSE(predictions2, test.data$rent_full),
  R2 = R2(predictions2, test.data$rent_full)
)


###############################################################
########### Polynomial-Regression model ######################

colnames(df_LM_kfold)

#create test and training set
set.seed(123)
training.samples <- df_LM_kfold$rent_full %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df_LM_kfold[training.samples, ]
test.data <- df_LM_kfold[-training.samples, ]

model <- lm(rent_full ~  I(dist_to_lake^2)+ I(Noise_max^2)+ I(Anteil_auslaend^2)+ 
              I(Avg_age^2)+ I(anteil_efh^2)+ I(Micro_rating^2)+ msregion +
              area + parking_indoor + rooms + wgh_avg_sonnenklasse_per_egid + Avg_size_household
            + avg_anzhl_geschosse +dist_to_4G + dist_to_5G + dist_to_haltst + dist_to_highway +
              dist_to_main_stat + dist_to_river + balcony + home_type + parking_outside + year_built + elevator
            + superm_pix_count_km2,   data = train.data)

model

predictions <- model %>% predict(test.data)
# Model performance
Poly_Reg <- data.frame(
  RMSE = RMSE(predictions, test.data$rent_full),
  R2 = R2(predictions, test.data$rent_full)
)

model2 <- lm(rent_full ~  I(dist_to_lake^2)+ I(Noise_max^2)+ I(Anteil_auslaend^2)+ 
               I(Avg_age^2)+ I(anteil_efh^2)+ I(Micro_rating^2), data = train.data)

predictions <- model2 %>% predict(test.data)
# Model performance
Poly_Reg2 <- data.frame(
  RMSE = RMSE(predictions, test.data$rent_full),
  R2 = R2(predictions, test.data$rent_full))

model3 <- lm(rent_full ~  I(dist_to_lake^2)+ I(Noise_max^2)+ I(Anteil_auslaend^2)+ 
               I(Avg_age^2)+ I(anteil_efh^2)+ I(Micro_rating^2)+ msregion +
               I(area^2) + I(parking_indoor^2) + I(rooms^2) + I(wgh_avg_sonnenklasse_per_egid^2) + I(Avg_size_household^2)
             + avg_anzhl_geschosse + I(dist_to_4G^2) + I(dist_to_5G^2) + I(dist_to_haltst^2) + I(dist_to_highway^2) +
               I(dist_to_main_stat^2) + I(dist_to_river^2) + I(balcony^2) + home_type + I(parking_outside^2) + I(year_built^2) + elevator
             + I(superm_pix_count_km2^2),   data = train.data)

predictions <- model3 %>% predict(test.data)
# Model performance
Poly_Reg3 <- data.frame(
  RMSE = RMSE(predictions, test.data$rent_full),
  R2 = R2(predictions, test.data$rent_full))




####### Lasso regression - k-fold CV ##################################################


set.seed(123)
training.samples <- df_LM_kfold$rent_full %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df_LM_kfold[training.samples, ]
test.data <- df_LM_kfold[-training.samples, ]

# Predictor variables
x <- model.matrix(rent_full~., train.data)[,-1]
# Outcome variable
y <- train.data$rent_full

#glmnet(x, y, alpha = 1, lambda = NULL)

#Lasso-Regression with Cross-validation
# Find the best lambda using cross-validation
set.seed(123) 
cv <- cv.glmnet(x, y, alpha = 1)
# Display the best lambda value
cv$lambda.min

# Fit the final model on the training data
model <- glmnet(x, y, alpha = 1, lambda = cv$lambda.min)
# Dsiplay regression coefficients
beta <- coef(model)

Lasso_Coeff <- as.data.frame(as.matrix(beta))

# Make predictions on the test data
x.test <- model.matrix(rent_full ~., test.data)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
# Model performance metrics
data.frame(
  RMSE = RMSE(predictions, test.data$rent_full),
  Rsquare = R2(predictions, test.data$rent_full)
)

#plotting lasso coef
Lasso=na.omit(train.data)
x=model.matrix(rent_full~.,Lasso)[,-1]
y=as.matrix(Lasso$rent_full)
lasso.mod =glmnet(x,y, alpha =1)
cf=coef(lasso.mod)
par(mfrow = c(1,1), mar = c(3.5,3.5,2,1), mgp = c(2, 0.6, 0), cex = 0.8, las = 1)
plot(lasso.mod, "lambda", label = TRUE)


####### Ridge - k-fold CV ###################################################

set.seed(123)
training.samples <- df_LM_kfold$rent_full %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df_LM_kfold[training.samples, ]
test.data <- df_LM_kfold[-training.samples, ]


# Predictor variables
x <- model.matrix(rent_full~., train.data)[,-1]
# Outcome variable
y <- train.data$rent_full

#glmnet(x, y, alpha = 1, lambda = NULL)

#Lasso-Regression with Cross-validation
# Find the best lambda using cross-validation
set.seed(123) 
cv <- cv.glmnet(x, y, alpha = 0)
# Display the best lambda value
cv$lambda.min

# Fit the final model on the training data
model <- glmnet(x, y, alpha = 0, lambda = cv$lambda.min)
# Dsiplay regression coefficients
ridge <- coef(model) ##### Welche Coef sind 0???

Ridge_Coeff <- as.data.frame(as.matrix(ridge))


# Make predictions on the test data
x.test <- model.matrix(rent_full ~., test.data)[,-1]
predictions <- model %>% predict(x.test) %>% as.vector()
# Model performance metrics
result <-data.frame(
  RMSE = RMSE(predictions, test.data$rent_full),
  Rsquare = R2(predictions, test.data$rent_full)
)

str(result)

Ridge=na.omit(train.data)
x=model.matrix(rent_full~.,Ridge)[,-1]
y=as.matrix(Ridge$rent_full)
ridge.mod =glmnet(x,y, alpha =0)
cf=coef(ridge.mod)
par(mfrow = c(1,1), mar = c(3.5,3.5,2,1), mgp = c(2, 0.6, 0), cex = 0.8, las = 1)
plot(ridge.mod, "lambda", label = TRUE)



####### RandomForest with CV ########################################################

df_rf <- subset(df, select = -c(GDENAMK, GDENR, msregion))
set.seed(123)
training.samples <- df_LM_kfold$rent_full %>%
  createDataPartition(p = 0.8, list = FALSE)
train.data  <- df_rf[training.samples, ]
test.data <- df_rf[-training.samples, ]

# very long computational time (~20h), but best model. 
# therefore commented out


# Fit the model on the training set
#set.seed(123)

#model_rf <- train(
#  rent_full ~., data = train.data, method = "rf",
#  trControl = trainControl("cv", number = 10))

#str(train.data)
# Best tuning parameter mtry
#model_rf$bestTune
# Make predictions on the test data
#predictions <- model_rf %>% predict(test.data)
#head(predictions)
# Compute the average prediction error RMSE
#RMSE(predictions, test.data$medv)


####### RandomForest  ########################################################


df_rf <- subset(df, select = -c(GDENAMK, GDENR, msregion, floors, avg_anzhl_geschosse))

str(df_rf)

# Create features and target
X <- df_rf[, -which(names(df_rf) == "rent_full")] 
y <- df_rf$rent_full

# Split data into training and test sets
index <- createDataPartition(y, p=0.75, list=FALSE)
X_train <- X[ index, ]
X_test <- X[-index, ]
y_train <- y[index]
y_test<-y[-index]

# Train the model 
set.seed(123)
regr <- randomForest(x = X_train, y = y_train, maxnodes = 5000, ntree = 140)

plot(regr)
regr

min_depth_frame <- min_depth_distribution(regr)
plot_min_depth_distribution(min_depth_frame)
plot_min_depth_distribution(min_depth_frame, mean_sample = "relevant_trees", k = 8)


# Make prediction
predictions <- predict(regr, X_test)

result <- X_test
result['rent_full'] <- y_test
result['prediction']<-  predictions

plot(result$rent_full, result$prediction)

ggplot(result, aes( x = seq.int(nrow(result)), y = rent_full)) +
  geom_point() + geom_point(aes(x = seq.int(nrow(result))), y = result$predicitions)

summ <- (result["rent_full"] - result["prediction"])^2
RMSE_rf <- sqrt(mean(summ$rent_full))
RMSE_rf

############Loop to find the best rf

#RMSE_rf <- c()

#for (i in seq(10,150,10)){
#  regr <- randomForest(x = X_train, y = y_train , maxnodes = 5000, ntree = i)
#  predictions <- predict(regr, X_test)
#  result <- X_test
#  result['rent_full'] <- y_test
#  result['prediction']<-  predictions
#  summ <- (result["rent_full"] - result["prediction"])^2
#  RMSE_rf[i] <- sqrt(mean(summ$rent_full))
#}

#RMSE_rf <- as.data.frame(RMSE_rf)
#RMSE_rf <- write.csv(RMSE_rf, "RMSE_rf.csv")
#plot(RMSE_rf)

############################

# rf graph
#min_depth_frame <- min_depth_distribution(regr)
#plot_min_depth_distribution(min_depth_frame)
#plot_min_depth_distribution(min_depth_frame, mean_sample = "relevant_trees", k = 8)

#plot_multi_way_importance(regr, size_measure = "no_of_nodes")

#plot_importance_ggpairs(regr)

###########################################################################
############################
# final prediction 


XX_test = as.data.frame(read.csv("X_test.csv"))
df1 = XX_test

# move ID to last position
df1 <- subset(df1, select=c(2:100,1))

na_count <- sapply(df1, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)



# Next, we delete all features with less than 5000 entries, after checking each
# of the features individually
df1 <- subset(df1, select = -c(which(na_count_start > 85001)))


df1 <- subset(df1, select = -c(Micro_rating_NoiseAndEmission,
                               Micro_rating_Accessibility, 
                               Micro_rating_DistrictAndArea, 
                               Micro_rating_SunAndView, 
                               Micro_rating_ServicesAndNature))


df1 <- subset(df1, select = -c(descr, newly_built, year, quarter_specific, address,
                               date, lat, lon, area_useable, cabletv, month, quarter_general, 
                               quarter_specific, year, apoth_pix_count_km2, 
                               restaur_pix_count_km2, avg_bauperiode, dist_to_school_1, 
                               geb_wohnnutz_total, wgh_avg_sonnenklasse_per_egid.1, kids_friendly, 
                               max_guar_down_speed))


discrete_vars = c("balcony", "elevator", "parking_indoor", "parking_outside")
df1[discrete_vars][is.na(df1[discrete_vars])] = 0


df1$floors[is.na(df1$floors)] = median(as.numeric(df$floors), na.rm = TRUE)
unique(df1$floors)
unique(df$floors)


df1$year_built[is.na(df1$year_built)] = 2016

df1$dist_to_lake[is.na(df1$dist_to_lake) | df1$dist_to_lake > 16000] = median(df$dist_to_lake, na.rm = TRUE)

df1$dist_to_main_stat[is.na(df1$dist_to_main_stat)] = median(df$dist_to_main_stat, na.rm = TRUE)

df1$Avg_size_household[is.na(df1$Avg_size_household)] = mean(df$Avg_size_household, na.rm = TRUE)

df1$Anteil_auslaend[is.na(df1$Anteil_auslaend) | df1$Anteil_auslaend == 1] = mean(df$Anteil_auslaend, na.rm = TRUE)

df1$Avg_age[df1$Avg_age < 23] = median(df$Avg_age, na.rm = TRUE)
df1$Avg_age[df1$Avg_age > 58] = median(df$Avg_age, na.rm = TRUE)
df1$Avg_age[is.na(df1$Avg_age)] = mean(df$Avg_age, na.rm = TRUE)

df1$avg_anzhl_geschosse[is.na(df1$avg_anzhl_geschosse)] = round(median(as.numeric(df$avg_anzhl_geschosse), na.rm = TRUE))

df1$anteil_efh[is.na(df1$anteil_efh)] = 0

df1$dist_to_haltst[is.na(df1$dist_to_haltst)] = median(df$dist_to_haltst, na.rm = TRUE)

df1$wgh_avg_sonnenklasse_per_egid[is.na(df1$wgh_avg_sonnenklasse_per_egid)] = mean(df$wgh_avg_sonnenklasse_per_egid, na.rm = TRUE)


sort(unique(df1$rooms))
df1$rooms[is.na(df1$rooms)] = round_any(df1$area[is.na(df1$rooms)] / mean_area_per_room, 0.5)
df1$rooms = round_any(df1$rooms, 0.5)
# still NAs in rooms
df1$rooms[is.na(df1$rooms)] = median(df$rooms, na.rm = TRUE)


df1$area[is.na(df1$area)] = round_any(df1$rooms[is.na(df1$area)] / mean_rooms_per_area, 1)

df1$avg_anzhl_geschosse = round_any(df1$avg_anzhl_geschosse, 1)

na_count <- sapply(df1, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)



# Make factors for relevant features that are characters
df1$home_type <- as.factor(df1$home_type)
df1$GDENR = as.factor(df1$GDENR)
df1$KTKZ = as.factor(df1$KTKZ)
df1$msregion = as.factor(df1$msregion)
df1$floors = as.factor(df1$floors)
df1$avg_anzhl_geschosse = as.factor(df1$avg_anzhl_geschosse)

# cleaning test set done.
########################


#predicting test set
####################

df_rf_test <- subset(df1, select = -c(GDENAMK, GDENR, msregion, floors, avg_anzhl_geschosse))
na_count <- sapply(df_rf_test, function(y) sum(length(which(is.na(y)))))

str(df_rf_test)
str(X_train)


our_pred <- predict(regr,  df_rf_test)
our_pred <- as.data.frame(our_pred)
predictions1 <- as.data.frame(round(our_pred$our_pred,1))
predictions1$ID <- df_rf_test$ID
predictions1 <- subset(predictions1, select = c(2,1))
colnames(predictions1 ) <- c("ID", "rent")

write.csv(predictions1, "Y_test.csv", row.names = FALSE)
