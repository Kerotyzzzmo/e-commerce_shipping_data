# e-commerce_shipping_data

## Introduction
The dataset used in this project is a widely recognized dataset from Kaggle, which has been extensively explored through explanatory data analysis. In this analysis, we aim to apply supervised machine learning algorithms like random forest and boosting to the dataset. We also include logistics regression and decision trees as the two baseline methods to establish comparisons. 

The performance of the models will be evaluated based on the testing error on an independent testing dataset. To fine-tune the models and optimize their performance, we will employ cross-validation techniques to identify the optimal values for the tuning parameters. Additionally, variable selection will be performed using a stepwise Bayesian information criterion (BIC) approach for the logistics regression method.

By comparing the performance of the random forest and boosting with the baseline models, we can assess their effectiveness in accurately predicting the target variable. The findings from this analysis will shed light on the most suitable model for this dataset and provide insights into the optimal tuning parameters and relevant variables that contribute to the model's performance. 

## Data Description 
This dataset contains a total of 10999 observations and 12 variables. The target variable used 1 or 0 to indicate if the shipment had reached on time. The data includes the following information:
* ID: Customer ID.  
* Warehouse Block: The company has one big warehouse divided into five blocks such as A, B, C, D, and E. 
* Mode of Shipment: Ship, Flight, or Road. 
* Customer care calls: The number of calls made from the enquiry for enquiry of the shipment.
* Customer rating: The company has rated every customer. 1 is the lowest (worst), and 5 is the highest (best). 
* Cost of the product: Cost of the product in US dollars. 
* Prior purchases: The number of prior purchases.
* Products importance: The company has categorized the product in various parameters such as low, medium, and high. 
* Gender: Male or Female.
* Discount offered: Discount offered on that specific product.
* Weight in gms: Shipment weight in grams. 
* Reached on time: It is the target variable, where 1 indicates that the shipment has NOT reached on time and 0 indicates that the shipment has reached on time.

## Methodology 
* Logistics Regression
* Decision Trees
* Random Forest
* Boosting

## Results 
* Logistics Regression = 0.6386
* Decision Tree	= 0.6703
* Random Forest	= 0.6682
* Boosting = 0.6519

After performing 10-fold cross-validation, it was observed that the decision tree method achieved the highest model accuracy. Both the random forest and boosting methods demonstrated similar results. Surprisingly, the logistic regression model had the lowest accuracy which indicated that it needs to do further feature engineering or variable selection. 

One limitation of logistic regression is its reliance on a single linear boundary. Unlike decision trees, which can divide the space into smaller areas using binary splits. Logistic regression only fits a single line to separate the space. This assumption of a linear relationship between dependent and independent variables may not always hold true, and it requires meeting certain assumptions to determine the model's goodness of fit. 

On the other hand, random forest is an ensemble method that combines multiple decision trees. While decision trees are easier to interpret and understand, random forest tends to be more computationally expensive due to the additional training time required. However, decision trees have a higher risk of overfitting. In contrast, random forest is more stable and reliable in its predictions. 

Boosting is another ensemble technique, and it introduces additional complexity with its numerous tuning parameters. Finding the optimal set of parameters to minimize testing errors can be challenging. However, when all hyperparameters are carefully tuned, boosting can outperform random forest in terms of performance.


