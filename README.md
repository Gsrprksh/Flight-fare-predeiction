
# Project Title
Flight Price Prediction


## Data Collection

The data have been collected from the Kaggle.
## pre processing steps

# first step

the data has been loaded into a dataframe. the shape of the data frame was (14781, 12).

the dataset contains different columns of class, airlines information, departure time, arrival time, duration, price etc..

most of the columns were categorical features.

the price column is the dependent column.

the data does not have any null values.

so no need of do any missing value operations.


# second step

# Exploratory Data Analysis

the data can be explored in various ways.

i have considered 4 main questions for the analysis.

those were as follows as below.

1.does the price varies as the airline changes?
2.How is the price affected when tickets are bought in just 1 or 2 days before departure?
3.Does ticket price change based on the departure time and arrival time?
4.How the price changes with change in Source and Destination?

The whole analysis was displayed in the jupyter notebook itself. that is flight fare prediction.py


## Encoding

After gone through the exploratory analysis i proceeded with identifying the 
main columns which can be make an impact on dependent feature if we make  changes in them.

I have used 'ORDINAL encoding' (Label Encoding) for the main features, as the ordinal encoding can be useful in ranking the values according to their weightage.



And, for remaining all i have used the one hot encoding.



## model building

for model building i have used linear regression model and random forest regressor model.

as the price feature was right scwed, i have normalize price column using log transformation. 
so that i can have the values in between the 0 to 1.

for model building in order to get the best r2 score i have iterate the model for 200 times to get best
number for random state parameter.


## results


finally i got the best results out of the model. the r2 score was around 0.88.
that means we got the best fit line as we thought of.

i have also checked with some test values, the mean absolute error was very less.
that shows the effectiveness of the model.



## flask application

I have used flask framework to build the api and web application. 

## deployment

Deployement was performed in the Heroku architecture. which is a platform as a service, where we can have runtime and server for ready to use.

