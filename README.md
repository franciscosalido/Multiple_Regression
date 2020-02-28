# Multiple_Regression
Multiple Regression examples of Machine Leaning

Introduction
In this article,I’m going to walk you through how to perform a multiple linear regression in python using the scikit-learn module.The scikit-learn module in python is basically used to perform machine learning tasks.It contains a number of machine learning models and algorithms that are very good and helpful to use.Among these models are the regression models that are used to perform both simple and multiple linear regression.
But one may ask what is machine learning and why do we need it? Machine learning is about extracting knowledge from data.Another way to view machine learning is training computers to learn from data using mathematical models. It is a research field at the intersection of statistics, artificial intelligence, and computer science and is also known as predictive analytics or statistical learning.Machine Learning can be classified into three main parts namely: Supervised Learning,Unsupervised Learning, and Reinforcement Learning.The most successful kinds of machine learning algorithms are those that automate decision-making processes by generalizing from known examples. In this setting, which is known as supervised learning.Supervised learning comprises of regression and classification so it is clearly seen that regression analysis falls into supervised machine learning.
Linear regression analysis,also known as linear modelling entails fitting a straight line,a plane or polynomial to a data.Like most of the machine learning algorithms,the goal of linear regression regression is to predict an output variable using other variables.Linear regression expresses the output variable or dependent variable as a function or a linear combination of the independent variables or the predictor variables.Linear regression is a widely used technique to model and understand real world phenomena.It is easy to use and understand intuitively.In simple linear regression,the model is just a straight line and for multiple regression,the model could either be a polynomial or a plane.But for this particular article,our main area of discussion is multiple linear regression.
The data we will be using for our regression analysis comprises of some technical specs of some cars.The data set is downloaded from UCI Machine Learning Repository.The UCI Machine Learning Repository is a machine learning repository that contains free data you can use for your machine learning projects.You can also go to my github repository for the data files and the python code written in a jupyter notebook. In this project,we try to predict the miles per gallon of a car using the multiple regression.Miles per gallon (mpg) of a car measures how far a car could go given one gallon of fuel.In every part of the world where the use of cars is very common,consumers sometimes consider the efficiency and fuel economy of the car they want to purchase before purchasing it.Everyone wants to buy a car that can travel far and consume less fuel.In this setting,let’s assume that we work for a firm that deals in cars and we have been given the task as data scientists to analyze this data and produce a good model that can predict/estimate the miles per gallon of a car with minimum error given other features of the car.
To begin with,we import the necessary modules we will be needing and they’re as follows.

Data preprocessing
In this part,all what we will do is to read in the data files,merge them and perform some data cleaning.Data cleansing or cleaning is a very important thing in the field of data analysis and machine learning.It requires detecting and correcting or removing inaccurate records from the data set.It improves your data quality and in doing so,increases overall productivity.When you clean your data,all outdated or incorrect information is gone leaving you with the highest quality information.


Now that we have our full data set,let’s describe what our columns or features mean to make it easier to understand them and know how useful they are in our analysis.
mpg — Mileage/Miles Per Gallon
cylinders — the power unit of the car where gasoline is turned into power
displacement — engine displacement of the car
horsepower — rate of the engine performance
weight — the weight of a car
acceleration — the acceleration of a car
model — model of the car
origin — the origin of the car
car — the name of the car
Now let’s print out the info of the data set.
cars.info()  # print the info of the data

we can see that the horsepower column is an object datatype and we will try to see what the odd value is in the horsepower column.We can go about this by getting the unique values in the horsepower column using pandas.unique() function and search for the odd value in the column.
# print the unique values in the horsepower column
cars.horsepower.unique()

We can see that the odd value is ‘?’ representing null so we now change it to NaN value and fill the spot with the mean horsepower.


We will drop the car column from the data set since we won’t be needing it.We also go ahead to check for duplicates and missing values in other columns.If there are any duplicated records,we will drop them and fill in the missing data with the mean value in case there is any.
# won't be needing the car column so we drop it
cars = cars.drop('car',axis=1)
# check for duplicates and null values
print('sum of duplicated values
{}\n'.format(cars.duplicated().sum()))
print('sum of null values: {}'.format(cars.isnull().sum()))

We can see from the output above that there are neither duplicated records nor missing data in our data set.Now we can say that our data is clean and ready to fit a model on,but we will first have to explore the data to find hidden patterns that will be of great help to our analysis.
Exploring the data
At this point,we are going to do a short and simple exploratory analysis on our data set to discover the relationships between variables,the distributions of the various features, and display some summary statistics of the data set.
# let's print the summary statistics of the data
display(cars.describe())

# let's visualize the distribution of the features of the cars
cars.hist(figsize=(12,8),bins=20)
plt.show()

What we can conclude from the histograms above is :
The acceleration of the cars in the data is normally distributed and the most of the cars have an acceleration of 15 meters per second squared.
Half of the total number of cars (51.3%) in the data have 4 cylinders.
Our output/dependent variable (mpg) is slightly skewed to the right.
We can also see that our variables are not on the same scale.
Let’s visualize the relationships between the Mileage Per Galon(mpg) of a car and the other features.
plt.figure(figsize=(10,6))sns.heatmap(cars.corr(),cmap=plt.cm.Reds,annot=True)
plt.title('Heatmap displaying the relationship between\nthe features of the data',
         fontsize=13)
plt.show()

Looking at the above correlation heatmap,we can conclude that;
We can see that there is a relationship between the mpg variable and the other variables and this satisfies the first assumption of Linear regression.
There is a strong negative correlation between the displacement,horsepower,weight,and cylinders.This implies that,as any one of those variables increases,the mpg decreases.
The displacement,horsepower,weight,and cylinders have a strong positive correlations between themselves and this violates the non-multi collinearity assumption of Linear regression.Multi-collinearity hinders the performance and accuracy of our regression model.To avoid this, we have to get rid of some of these variables by doing feature selection.
The other variables.ie.acceleration,model and origin are not highly correlated with each other. We can also check for multi-collinearity using the variance inflation factor.A variable/feature affected by multi-collinearity will have a value greater than 5 when we print out the series from the variance inflation factor
Another way of checking the multi-collinearity is by using the variance inflation factor.If a variable has a variance inflation factor greater than 5,then it is associated with multi-collinearity.We will use the variance_inflation_factor() of statsmodels to perform this task and the code is as shown below.
X1 = sm.tools.add_constant(cars)
# calculate the VIF and make the results a series.
series1 = pd.Series([variance_inflation_factor(X1.values,i) for i in range(X1.shape[1])],index=X1.columns)
print('Series before feature selection: \n\n{}\n'.format(series1))

We can see that there is a problem of multi-collinearity in our data since some of the variables have a variance inflation factor greater than 5.And we can also see clearly that the displacement,horsepower,weight,and cylinders have a strong positive correlations between themselves and they are the cause of the multi-collinearity as shown in the correlation heatmap above.To avoid this, we take out those features from our data and compute the variance inflation factors of the remaining variables and check if multi-collinearity still exists.
# Let's drop the columns that highly correlate with each other
newcars = cars.drop(['cylinders','displacement','weight'],axis=1)
# Let's do the variance inflation factor method again after doing a feature selection
#to see if there's still multi-collinearity.
X2 = sm.tools.add_constant(newcars)
series2 = pd.Series([variance_inflation_factor(X2.values,i) for i in range(X2.shape[1])],index=X2.columns)
print('Series after feature selection: \n\n{}'.format(series2))

Great, we have gotten rid of the multi-collinearity as the remaining variables have a variance inflation factor less than 5.Now we have gotten enough information from our data and it’s time to fit train and fit a model on it and start making some predictions.
Training the regression model
This is the part where we start training the regression models we imported earlier.Here, we do not train only one model.But we train as many models as we can as model accuracy is what were after.We want to end up with the model that predicts well and gives minimum error.We will split our dataset into two parts.ie. training data and testing data using the train_test_split() function of sklearn.model_selection.Since the variables are not of the same scale,we will scale them using the preprocessing.scale() function from sklearn.Scaling the variables is only necessary for the linear,ridge and lasso regression models as these models penalize coefficients.After scaling the feature or predictor variables,we will therefore go ahead to fit our LinearRegression model on the data and assess the model to see how accurate it is.


We can see from the above output that the LinearRegression model fits on the training data 75.5% and 72.7% on the test set.With this model,we do not have a problem of over-fitting or under-fitting but the accuracy of the model isn’t satisfactory so we go ahead and fit a Ridge model on the data to see if we can increase the accuracy and minimize the mean squared error.


Looks like the Ridge model is no different from the LinearRegression model we first fit.Let’s try to tune the hyper parameters to see if we can make a significant change in the accuracy and minimize the MSE.In doing this,we will perform a grid search cross validation to search for the best parameters using the GridSearchCV() function from sklearn.model_selection


Nothing much from Ridge regression,we move on to fitting a Lasso regression model and straight away perform a grid search for the best parameters.


I guess it’s clear to us now that the LinearRegression,Ridge, and Lasso are giving us a non-satisfactory model accuracy and a mean squared error so we move on to the ensemble methods for our regression.The most common ensemble methods we will use are the DecisionTree,RandomForest and GradientBoosting.Instead of first fitting the models with single parameters and scoring them,we will straight away begin with the grid search for the best parameters and score the models.Let’s begin with the DecisionTreeRegressor and tune its parameters.


From the output above,we can clearly see that the DecisionTreeRegressor has an accuracy of 79% and a mean squared error of 11.4 which is better than the LinearRegression,Ridge and Lasso models.But it looks like the DecisionTreeRegressor is slightly over-fitting as its prediction accuracy on the training data is 86.6% and on the test data is 79%. Let’s consider the RandomForestRegressor model to see if we can still get a higher accuracy,minimized error,and a generalized model.


The RandomForestRegressor is doing great with reducing the mean squared error but also over-fitting the data as its prediction accuracy on the training data is 94% and on the test data is 80.5%. Let’s consider the GradientBoostingRegressor model to see if we can still get a higher accuracy,minimized error,and a generalized model.


Looks like this model is not too over-fitted and it has low mean squared error which when taken the square root of gives 2.98.This tells us that the average distance from the actual values and predicted values is 2.98 which is better.We will now try to make predictions and see how well our model predicts.We will visualize the actual mpg values recorded and the mpg values predicted by our model to see how close our predictions are to the actual values.


We can see from the above scatter plot that our model made a good predictions as the values of the actual mpg and the predicted mpg are very close to each other.We can say that we have succeeded in training a model that predicts the Mileage Per Gallon (mpg) of a car given the acceleration,model,origin and the horsepower of a car.
Even though we could have continued to train other models like the Adaboost and XGboost which could have given a better accuracy and minimized error as compared to our final GradientBoosting model but I choose to end here since this whole article was just to show you how to train a multiple regression model to make estimations/predictions.I hope to go further next time and thank you for taking your precious time to read this article.The whole project files and code can be accessed at my github account.Thanks again and see you next time.
