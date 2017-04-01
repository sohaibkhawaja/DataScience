# Syracuse University : iSchool - Data Science Club
# Author : Sohaib Khawaja - SUID 3956085181 - arkitekt
# Kernel : The Titanic Survival - ML - Random Forests - Tutorial


# Load packages
library(mice)
library(dplyr)
library(randomForest)
library(arulesViz) # visualizing association rules
library(ggplot2) # visualization
library(ggthemes) # visualization
library(scales) # visualization

setwd("C:/DataScience/SU-DSC/Titanic/Data")


# Load data
train <- read.csv("train.csv", stringsAsFactors = F)
head(t.train)

test  <- read.csv("test.csv", stringsAsFactors = F)
head(t.test)

t  <- bind_rows(train, test) # bind training & test data
tail(t)

t[890,]
t[980,]
str(t)

#Variable Name | Description
#--------------|-------------
#Survived      | Survived (1) or died (0)
#Pclass        | Passenger's class
#Name          | Passenger's name
#Sex           | Passenger's sex
#Age           | Passenger's age
#SibSp         | Number of siblings/spouses aboard
#Parch         | Number of parents/children aboard
#Ticket        | Ticket number
#Fare          | Fare
#Cabin         | Cabin
#Embarked      | Port of embarkation


## Data cleaning and feature engineering

## Passenger's name

# Grab title from passenger names
t$Title <- gsub('(.*, )|(\\..*)', '', t$Name)

# Show title counts by sex
table(t$Sex, t$Title)

# low frequency Titles can be combined as "rare" level

rare_title <- c('Dona', 'Lady', 'the Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
t$Title[t$Title == 'Mlle']        <- 'Miss' 
t$Title[t$Title == 'Ms']          <- 'Miss'
t$Title[t$Title == 'Mme']         <- 'Mrs' 
t$Title[t$Title %in% rare_title]  <- 'Rare Title'

# Show title counts by Sex again
table(t$Sex, t$Title)

# Finally, grab surname from passenger name
t$Surname <- sapply(t$Name, function(x) strsplit(x, split = '[,.]')[[1]][1])


cat(paste('We have ', nlevels(factor(t$Surname)), ' unique Surnames onboard Titanic.'))


## Do families sink or swim together?

# Now that we've taken care of splitting passenger name into some new variables, we can take it a step
# further and make some new family variables. 
# First we're going to make a **family size** variable based on number of siblings/spouse(s) 
# (maybe someone has more than one spouse?) and number of children/parents. 

# Create a family size variable including the passenger themselves
t$Fsize <- t$SibSp + t$Parch + 1

# Create a family variable 
t$Family <- paste(t$Surname, t$Fsize, sep='_')

#What does our family size variable look like? To help us understand how it may relate to survival,
#let's plot it among the training data.

par(mar=c(2, 2, 2, 1), mfrow=c(1,1)) # plot margins and area

# Use ggplot2 to visualize the relationship between family size & survival
ggplot(t, aes(x = Fsize, fill = factor(Survived))) +
  geom_bar(stat='count', position='dodge') +
  scale_x_continuous(breaks=c(1:11)) +
  labs(x = 'Family Size') + theme_few()

#We can see that there' s a survival penalty to singletons and those with family sizes above 4. 
#We can collapse this variable into three levels which will be helpful since there are comparatively
#fewer large families. Let's create a **discretized family size** variable.

t$FsizeD[t$Fsize == 1] <- 'singleton'
t$FsizeD[t$Fsize < 5 & t$Fsize > 1] <- 'small'
t$FsizeD[t$Fsize > 4] <- 'large'

# Show family size by survival using a mosaic plot
mosaicplot(table(t$FsizeD, t$Survived), main='Family Size by Survival', shade=TRUE)

#The mosaic plot shows that we preserve our rule that there's a survival penalty among singletons and
#large families, but a benefit for passengers in small families. I want to do something further with our age variable, but `r sum(is.na(full$Age))` rows have missing age values, so we will have to wait until after we address missingness.

## Treat a few more variables ...

# **passenger cabin** variable including about their **deck**. Let's take a look.

# This variable appears to have a lot of missing values
t$Cabin[1:50]

# The first character is the deck. For example:
strsplit(t$Cabin[2], NULL)[[1]]

# Create a Deck variable. Get passenger deck A - F:
t$Deck <- factor(sapply(t$Cabin, function(x) strsplit(x, NULL)[[1]][1]))


# There's more that likely could be done here including looking into cabins with multiple rooms listed 
# (e.g., row 28: "C23 C25 C27"), but given the sparseness of the column we'll stop here.

# Missingness - Managing Empties and NA values  
# Now we're ready to start exploring missing data and rectifying it through imputation. 
# There are a number of different ways we could go about doing this. Given the small size of the dataset,
# we probably should not opt for deleting either entire observations (rows) or variables (columns)
# containing missing values. We're left with the option of either replacing missing values with a sensible
# values given the distribution of the data, e.g., the mean, median or mode. 
# Finally, we could go with prediction. We'll use both of the two latter methods and use visualization
# for guidance.

## Sensible value imputation

# Passengers 62 and 830 are missing Embarkment
t[t$Embarked=="", ] 
t[t$Ticket==113572, ] 

# Passengers 62 and 830 also seem to be having the same ticket number, but let's fix the empties first.
cat(paste('We will infer their values for **embarkment** based on present data that we can imagine may be relevant: **passenger class** and **fare**. We see that they paid $', t[62, 'Fare'][[1]][1], 'and $', t[830, 'Fare'][[1]][1], 'respectively and their classes are', t[62, 'Pclass'][[1]][1], 'and', t[830, 'Pclass'][[1]][1], '. So from where did they embark?'))

# Get rid of our missing passenger IDs
embark_fare <- t %>% filter(PassengerId != 62 & PassengerId != 830)

# Use ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(embark_fare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
             colour='red', linetype='dashed', lwd=2) +
  scale_y_continuous(labels=dollar_format()) +  theme_few()

# The median fare for a first class passenger departing from Charbourg ('C') coincides nicely with the $80 paid
# by our embarkment-deficient passengers. I think we can safely replace the NA values with 'C'.

# Since their fare was $80 for 1st class, they most likely embarked from 'C'
t$Embarked[c(62, 830)] <- 'C'
t[t$Ticket==113572, ] # confirm that Embarked has 'C' for our 2 passengers 

# We're close to fixing the handful of NA values here and there. 
# Passenger on row 1044 has an NA Fare value.


# Show row 1044
t[1044, ]

# This is a third class passenger who departed from Southampton ('S'). 
# Let's visualize Fares among all others sharing their class and embarkment 
#(n = `r nrow(t[full$Pclass == '3' & t$Embarked == 'S', ]) - 1`).

ggplot(t[t$Pclass == '3' & t$Embarked == 'S', ], 
       aes(x = Fare)) +
  geom_density(fill = '#99d6ff', alpha=0.4) + 
  geom_vline(aes(xintercept=median(Fare, na.rm=T)),
             colour='red', linetype='dashed', lwd=1) +
  scale_x_continuous() +  theme_few()

# From this visualization, it seems quite reasonable to replace the NA Fare value with median for their
# class and embarkment which is { median(t[t$Pclass == '3' & t$Embarked == 'S', ]$Fare, na.rm = TRUE) }

# Replace missing fare value with median fare for class/embarkment
t$Fare[1044] <- median(t[t$Pclass == '3' & t$Embarked == 'S', ]$Fare, na.rm = TRUE)

## Predictive imputation
# Finally, as we noted earlier, there are quite a few missing **Age** values in our data. 
# We are going to get a bit more fancy in imputing missing age values. Why? Because we can. 
# We will create a model predicting ages based on other variables.

# Show number of missing Age values
sum(is.na(t$Age))

#We could definitely use `rpart` (recursive partitioning for regression) to predict missing ages, 
#You can read more about multiple imputation using chained equations in r 
# [here](http://www.jstatsoft.org/article/view/v045i03/v45i03.pdf) (PDF). 
# Since we haven't done it yet, I'll first factorize the factor variables and then perform mice imputation.


# Make variables factors into factors
fac_vars <- c('PassengerId','Pclass','Sex','Embarked','Title','Surname','Family','FsizeD')

t[fac_vars] <- lapply(t[fac_vars], function(x) as.factor(x))



# Set a random seed
set.seed(11)

# Perform mice imputation, excluding certain less-than-useful variables:
mice_mod <- mice(t[, !names(t) %in% c('PassengerId','Name','Ticket','Cabin','Family','Surname','Survived')], method='rf')

# Save the complete output 
mice_output <- complete(mice_mod)


# Let's compare the results we get with the original distribution of passenger ages to ensure that nothing has gone completely awry.

# Plot age distributions
par(mfrow=c(1,2))
hist(t$Age, freq=F, main='Age: Original Data', 
     col='blue', ylim=c(0,0.04), border = "white")
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='dodgerblue', ylim=c(0,0.04), border="white")

# Things look good, so let's replace our age vector in the original data with the output from the `mice` model.

# Replace Age variable from the mice model.
t$Age <- mice_output$Age

# Confirm number of missing Age values
sum(is.na(t$Age)) # very cool


# We've finished imputing values for all variables that we care about for now! 
# Now that we have a complete Age variable, there are just a few finishing touches I'd like to make. 
# We can use Age to do just a bit more feature engineering ...


## Feature Engineering: Round 2

# Now that we know everyone's age, we can create a couple of new age-dependent variables: 
# **Child** and **Mother**. A child will simply be someone under 18 years of age and a mother is 
# a passenger who is
# 1) female, 
# 2) is over 16, (I doubt that in 1912 first-time mothers in UK were above 18!)
# 3) has more than 0 children (no kidding!), and 
# 4) does not have the title 'Miss'.

# First we'll look at the relationship between age & survival
ggplot(t[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram() + 
  # I include Sex since we know (a priori) it's a significant predictor
  facet_grid(.~Sex) + 
  theme_few()

# Create the column Age Adult/Child, and indicate whether child or adult
t$AgeAC[t$Age < 16] <- 'Child'
t$AgeAC[t$Age >= 16] <- 'Adult'

# Show counts
table(t$AgeAC, t$Survived)

# Looks like being a child doesn't hurt, but it's not going to necessarily save you either! 
# We will finish off our feature engineering by creating the **Mother** variable. 
# Maybe we can hope that mothers are more likely to have survived on the Titanic.

# Adding Mother variable
t$Mother <- 'Not Mother'
t$Mother[t$Sex == 'female' & t$Parch > 0 & t$Age > 16 & t$Title != 'Miss'] <- 'Mother'

# Show counts
table(t$Mother, t$Survived)

# Finish by factorizing our two new factor variables
t$AgeAC  <- factor(t$AgeAC)
t$Mother <- factor(t$Mother)


# data seems pretty complete and clean.  Confirm just in case.
md.pattern(t)

# Wow! We have finally finished treating all of the relevant missing values in the Titanic dataset 
# which has included some fancy imputation with `mice`. We have also successfully created several new 
# variables which we hope will help us build a model which reliably predicts survival. 

# Prediction
# At last we're ready to predict who survives among passengers of the Titanic based on variables that we carefully curated and treated for missing values. For this, we will rely on the `randomForest` classification algorithm; we spent all that time on imputation, after all.

## Split into training & test sets

#Our first step is to split the data back into the original test and training sets.

# Split the data back into a train set and a test set
t.train <- t[1:891,]
t.test <- t[892:1309,]



## Building the model 

#We then build our model using `randomForest` on the training set.

# Set a random seed
set.seed(1942)

# Build the model (note: not all possible variables are used)
rf_m <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + FsizeD + AgeAC + Mother,
                         data = t.train)


rf_m <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                       Fare + Embarked + Title + FsizeD + AgeAC,
                     data = t.train)


rf_m <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                       Fare + Embarked + Title + FsizeD ,
                     data = t.train)

# Show model error
par(mfrow=c(1,1))
plot(rf_m, ylim=c(0,0.36))
legend('topright', colnames(rf_m$err.rate), col=1:3, fill=1:3)


# The black line shows the overall error rate which falls below 20%. 
# The red and green lines show the error rate for 'died' and 'survived' respectively. 
# We can see that right now we're much more successful predicting death than we are survival. 

## Variable importance

#Let's look at relative variable importance by plotting the mean decrease in Gini calculated across all trees.

# Get importance
importance    <- importance(rf_m)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))


# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_few()


# Whoa, glad we made our title variable! 
# It has the highest relative importance out of all of our predictor variables. 
# I think I'm most surprised to see that passenger class fell to `r rankImportance[rankImportance$Variable == 'Pclass', ]$Rank`, but maybe that's just bias coming from watching the movie Titanic too many times as a kid.

## Prediction!

# We're ready for the final step --- making our prediction! 
# When we finish here, we could iterate through the preceding steps making tweaks as we go or fit the data using different models or use different combinations of variables to achieve better predictions. But this is a good starting (and stopping) point for me now.


# Predict using the test set
prediction <- predict(rf_m, t.test)

# Save the solution to a dataframe with two columns: PassengerId and Survived (prediction)
solution <- data.frame(PassengerID = t.test$PassengerId, Survived = prediction)

# Write the solution to file
write.csv(solution, file = 'titanic-survival-arkitekt.csv', row.names = F)

# Conclusion

#Thank you for taking the time to read through my first exploration of a Kaggle dataset. I look forward to doing more. Again, this newbie welcomes comments and suggestions!

---


# 1)	Compute the percentage of people that survived.   

onboard <- nrow(t)
SurYes <- nrow(t[t$Survived == "Yes",])
SurYesPer <- nrow(t[t$Survived == "Yes",])/onboard*100

cat("Number of pepole onboard Titanic :", onboard
  , "\nNumber of pepole who survived    :", SurYes
  , "\nPercentage of people who survived:", SurYesPer)


# 2)	Compute the percentage of people that were children

onboardChild <- nrow(t[t$Age == "Child",])
onboardChildPer <- nrow(t[t$Age == "Child",])/onboard*100

cat("Number of children onboard Titanic:", onboardChild
    , "\nPercentage of people onboard;Child:", onboardChildPer)

# 3)	Compute the percentage of people that were female

onboardFemale <- nrow(t[t$Sex == "Female",])
onboardFemalePer <- nrow(t[t$Sex == "Female",])/onboard*100

cat("Number of females onboard Titanic  :", onboardFemale
    , "\nPercentage of people onboard;Female:", onboardFemalePer)


# 4)	Finally, compute the percentage of people that were in first class

onboardFirst <- nrow(t[t$Class == "1st",])
onboardFirstPer <- nrow(t[t$Class == "1st",])/onboard*100

cat("Number of people in 1st Class onboard Titanic    :", onboardFirst
    , "\nPercentage of people in 1st Class onboard Titanic:", onboardFirstPer)


# Step 2: More Descriptive Stats
# 1)	What percentage of children survived?


SurChild <- nrow(t[t$Survived == "Yes" & t$Age == "Child",])

SurChildPer <- nrow(t[t$Survived == "Yes" & t$Age == "Child",])/onboardChild*100
PerSurChild <- nrow(t[t$Survived == "Yes" & t$Age == "Child",])/SurYes*100

cat(SurChildPer," percent of children onboard Titanic survived, out of total ",onboardChild
    ,"\n",PerSurChild," percent of survivors were children of the total ",SurYes)


# 2)	What percentage of female survived?

SurFemale <- nrow(t[t$Survived == "Yes" & t$Sex == "Female",])

SurFemalePer <- nrow(t[t$Survived == "Yes" & t$Sex == "Female",])/onboardFemale*100
PerSurFemale <- nrow(t[t$Survived == "Yes" & t$Sex == "Female",])/SurYes*100

cat(SurFemalePer," percent of females onboard Titanic survived, out of total ",onboardFemale
    ,"\n",PerSurFemale," percent of survivors were females of the total ",SurYes)


# 3)	What percentage of first class people survived?

Sur1Class <- nrow(t[t$Survived == "Yes" & t$Class == "1st",])

Sur1ClassPer <- nrow(t[t$Survived == "Yes" & t$Class == "1st",])/onboardFirst*100
PerSur1Class <- nrow(t[t$Survived == "Yes" & t$Class == "1st",])/SurYes*100

cat(Sur1ClassPer," percent of people in 1st Class onboard Titanic survived, out of total ",onboardFirst
    ,"\n",PerSur1Class," percent of survivors were from 1st Class of the total ",SurYes)

# 4)	What percentage of 3rd class people survived?

onboardThird <- nrow(t[t$Class == "3rd",])

Sur3Class <- nrow(t[t$Survived == "Yes" & t$Class == "3rd",])

Sur3ClassPer <- nrow(t[t$Survived == "Yes" & t$Class == "3rd",])/onboardThird*100
PerSur3Class <- nrow(t[t$Survived == "Yes" & t$Class == "3rd",])/SurYes*100

cat(Sur3ClassPer," percent of people in 3rd Class onboard Titanic survived, out of total ",onboardThird
    ,"\n",PerSur3Class," percent of survivors were from 3rd Class of the total ",SurYes)

t

# Step 3: Writing a Function
# 1)	Write a function that returns the a new dataframe of people that satisfy the specified criteria of
#     sex, age, class and survived as parameters

myPeople <- function(df, Class, Sex, Age, Survived){
  return(df[df$Class == Class & df$Sex == Sex & df$Age == Age & df$Survived == Survived,])
}

myPeople(t, '3rd', 'Female', 'Child', 'No')

# 2)	Write a function, using the previous function, that calculates the percentage (who lives, who dies) for a specified (parameters) of age, class and sex.

livORdie <- function(df, Class, Sex, Age)
  {
  lived <- nrow(myPeople(df, Class, Sex, Age, 'Yes'))
  died <- nrow(myPeople(df, Class, Sex, Age, 'No'))
  percentage <- lived/(lived+died)*100
  return(percentage)
}

# 3)	Use the function to compare age & 3rd class male survival rates

cat("3rd class male survival rates \nAdult : ", livORdie(t, '3rd', 'Male', 'Adult')
    ,"\nChild : ",livORdie(t, '3rd', 'Male', 'Child'))

# 4)	Use the function to compare age & 1st class female survival rates

cat("1st class female survival rates \nAdult : ", livORdie(t, '1st', 'Female', 'Adult')
    ,"\nChild : ",livORdie(t, '1st', 'Female', 'Child'))

# Step 4: Use aRules
# 1)	Use arules to calculate some rules (clusters) for the titanic dataset

t.rules <- apriori(t)
inspect(t.rules)

# 2)	Visualize the results
plot(t.rules, cex = 1.5, alpha = 0.5)

# 3)	Pick the 3 most interesting & useful rules.
ruleset <- apriori(t, parameter=list(support=0.005, confidence=0.35)) 
inspect(ruleset)

plot(ruleset, cex = 2, alpha = 0.5)

newrules <- ruleset[quality(ruleset)$lift < 3.9 & quality(ruleset)$lift > 3.3]
inspect(newrules)

# 4)	How does this compare to the descriptive analysis we did on the same dataset? 

cat("It provides a bit more detail and it helps with big datasets. \nDescriptive analysis is not always possible if your dataset has non numeric or categorical data.")

