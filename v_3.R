###

library(tidyverse) # some useful functions, such as read_csv are included in this package.
library(psych) # some useful functions, such as describe are included in this package
library(corrplot) # to plot correlations
library(margins) # to calculate marginal effects after the Logit models.
library(ggplot2)
library(Hmisc)
library(ggcorrplot)
library(hot.deck)
library(dplyr)
library(plyr)



getwd()
setwd("C:/Users/kraft/OneDrive/Desktop/Bewerbung/UL/3_HS_2021/Supervised Learning/Final Project")
list.files()
list.files(pattern=".csv")



data_basis <- as.data.frame(read.csv("training.csv"))
df <- data_basis
dim(df)
str(df)
#View(df)


# We consider the extent of the dataset as to be too vast. 
# Therefore, we use various approaches to reduce first, the number of observations and, 
# second, the number of variables.


# Counting for features with extreme high number of NAs. These will later be
# excluded, since we can assume missing data in some cases rather than scarce features
na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)


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


# We consider the vif-test to test for multi-collinearity.
# Therefore, we fit an easy linear model.
model_test_mc = lm(rent_full ~ area + as.factor(floors), df)
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


# Now we can start with the models. 










na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)


# substitute area NAs
# area

summary(area)
boxplot(area)
hist(area, breaks = 100)

# we use the mean to replace the NAs. Not really relevant as it's only 21 observations, and the spread is not very large.
df$wgh_avg_sonnenklasse_per_egid[is.na(df$wgh_avg_sonnenklasse_per_egid)] = mean(df$wgh_avg_sonnenklasse_per_egid, na.rm = TRUE)







na_count <-sapply(df, function(y) sum(length(which(is.na(y)))))
na_count <- data.frame(na_count)






# Make factors for relevant features that are characters
df$home_type <- as.factor((df$home_type))
df$GDENR = as.factor(df$GDENR)
df$msregion = as.factor(df$msregion)
df$floors = as.factor(df$floors)

# Descriptive analysis
########################

# Examine histograms for all variables
par(mfrow=c(3,3))
hist.data.frame(df)


# The easiest way for the detection of multicollinearity is to examine 
# the correlation between each pair of explanatory variables. 
# If two of the variables are highly correlated, then this may the 
# possible source of multicollinearity.


# correlation
# Take a first glance at correlations between variables
# this plot has been used before for determining variables to be deleted due to multicorrelations
cor(df[, unlist(lapply(df, is.numeric))], use = "pairwise.complete.obs")

ggcorrplot(cor(df[, unlist(lapply(df, is.numeric))], use = "pairwise.complete.obs"), method = "circle")


# more descriptive statistics
########################

# Plot specific relations for further examination
# area
par(mfrow=c(1,1))

ggplot(df, aes(x= area, y= rent_full), use = "pairwise.complete.obs") + 
  geom_point() +
  geom_smooth()


ggplot(df, aes(x= Micro_rating, y= rent_full), use = "pairwise.complete.obs") + 
  geom_point() +
  geom_smooth()

plot(df$msregion, df$rent_full, xlab = "msregion", ylab ="rent", varwidth = T, col = c(2:106))




#models

#first naive model
attach(df)


plot(rent_full,area)
lm.fit=lm(rent_full~area+Micro_rating+dist_to_4G)
summary(lm.fit)

predict(lm.fit, df[1:10000,], interval="confidence")
predict(lm.fit, df[1:10000,], interval="prediction")



# general linear model

glm.fit = glm(rent_full ~ .-GDENAMK-GDENR-KTKZ, data = df)
summary(glm.fit)


library("caret")

set.seed(107)
train_set = createDataPartition(
  y = df$rent_full,
  # the outcome data are needed
  p = .8,
  # The percentage of data in the
  # training set
  list = FALSE # The format of the results
)

training = df[train_set,]
training = training[, -c(1:3,7)]
training[is.na(training)] <- 0
testing  = df[-train_set,]
testing = testing[, -c(1:3,7)]
testing[is.na(testing)] <- 0

glm_mod = train(
  form = rent_full ~ .,
  data = training,
  #trControl = trainControl(method = "cv", number = 5),
  method = "glm",
  family = "gaussian", na.action = na.pass
)

is.na(summary(glm_mod)$coefficients)

summary(glm_mod)
glm_mod$finalModel
glm_mod$results

predicted = predict.lm(glm_mod, newdata = testing)
mean((testing$rent_full-predicted)^2)
predicted



#next model
.....
.....





# training / test split of subset

train1 = df[sample(nrow(df), 10000), ]
test1 = df[89000:90000,]


# first model

lm.fit = lm(rent_full~., data=train1)
summary(lm.fit)
par(mfrow=c(2,2))
plot(lm.fit)


lm.fit1 = lm(rent_full~Micro_rating, data = train1)
summary(lm.fit1)
plot(lm.fit1)

lm.fit2 = lm(rent_full~area+Micro_rating+parking_indoor*parking_outside, data = train1)
summary(lm.fit2)
par(mfrow=c(2,2))
plot(lm.fit2)


glm.fit = glm(Direction ~ Lag2, data = Weekly, family = binomial, subset = train)
glm.probs = predict(glm.fit, test, type = "response")
glm.pred = rep("Down", length(glm.probs))
glm.pred[glm.probs > 0.5] = "Up"
Direction.test = Direction[!train]
table(glm.pred, Direction.test)
mean(glm.pred == Direction.test)

#library(caret)
inTrain = createDataPartition(
  y = df$rent_full,
  # the outcome data are needed
  p = .75,
  # The percentage of data in the
  # training set
  list = FALSE # The format of the results
)


training = df[inTrain,]
testing  = df[-inTrain,]

glm_mod = train(
  form = rent_full ~ .,
  data = training,
  #trControl = trainControl(method = "cv", number = 5),
  method = "glm",
  family = "binomial"
)

summary(glm_mod)
glm_mod$finalModel
glm_mod$results




# from exercise 8

results.knn.1 <- train(purchased ~ ., 
                       data = d.training, 
                       method = "knn", 
                       preProcess = "scale",
                       tuneGrid = data.frame(k=5), 
                       trControl = trainControl("none"))

# 3. Calculate in sample and out of sample performance metrics
# ------------------------------------------------------------------------------

# (b) Out of sample performance metrics
d.test$prediction.knn5.class <- predict(object = results.knn.1,
                                        newdata = d.test)
tab.b <- table(d.test$prediction.knn5.class, d.test$purchased)
confusionMatrix(tab.b, positive = "purchased")


# Leave-p-out cross validation
set.seed(123)

results.knn.3 <- train(purchased ~ ., 
                       data = d.training, 
                       method = "knn", 
                       tuneGrid = data.frame(k=5), 
                       trControl = trainControl(method = "LGOCV", p = 0.9,
                                                savePredictions = T))


# K-Fold cross validation
set.seed(123)
results.knn.4 <- train(purchased ~ ., 
                       data = d.training, 
                       method = "knn", 
                       tuneGrid = data.frame(k=5), 
                       trControl = trainControl(method = "cv",
                                                savePredictions = T))




