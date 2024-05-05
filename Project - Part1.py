
# =============================================================================
# PREDICTING PRICE OF USED CARS 
# =============================================================================

# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# READING CSV FILE
# =============================================================================
example_data = pd.read_csv(r"E:\Project\Used_Bikes.csv")

# Copy dataframe
example = example_data.copy()

#Summarise the data
example.info()

# =============================================================================
# Analysis on price feature
# =============================================================================
# To display maximum number of columns
pd.set_option("display.max_columns",100)
pd.set_option("display.max_rows",4000)

# Checking the unique values in "price" column
unique_prices = example.price.unique()

print(unique_prices)

#To get the sample idea on data
example.price.sample(5)

# Checking the datatype of price column
example.price.dtypes

def extract_first(value):
    
    if isinstance(value,str) and " " in value:
        return int(value.split()[0])
    else:
        return int(value)

example['price'] = example['price'].apply(extract_first)

example['price'].dtypes

example.info()

# =============================================================================
# Analysis on kilometer feature
# =============================================================================

#To count the number of values 
pd.value_counts(example.km_driven)

#Finding the unique values
np.unique(example.km_driven)

#Checking the datatype
example.km_driven.dtypes
# example["km_driven"].astype("int64")

#Finding no.of nulls in all columns
example.isna().sum()
# Finding rows having missing values in atleast on e column
missing = example[example.isna().any(axis = 1)]
#Dropping null values from km_driven columns
example = example.dropna(subset = ["km_driven"])
example.km_driven.dtypes
np.unique(example.km_driven)
#To Check dataset that if changes reflected or not
example.info()

#Removing the "brand" and "age" columns
column_name = ["brand","age"]
example.drop(column_name,axis = 1,inplace = True)

example.info()

# Converting the dataframe to csv file
example.to_csv(r"E:\Project\new_bike_data.csv",index = False)

new_data = example.copy()

# To get the current working directory
# Suppose if you won't give the path it will default store in 'C:\\Users\\vishn'
# import os
# os.getcwd()

# Replacing unnecessary character "<" in km_driven column
# new_data.km_driven = new_data.km_driven.str.replace("<","")
new_data.info()
# Making column value to null which contains words such as "Mileage" and "Yes  "
new_data.km_driven = np.where(new_data.km_driven.str.contains("Mileage"),np.nan,new_data["km_driven"])
new_data.km_driven = np.where(new_data.km_driven.str.contains("Yes  "),np.nan,new_data.km_driven)
new_data.info()
# Converting all values to numeric 
new_data["km_driven"] = pd.to_numeric(new_data.km_driven)
new_data.km_driven.dtypes
new_data.to_csv(r"E:\Project\checking_data.csv",index = False)

# example.km_driven.astype("int64")
example.km_driven.isna().sum()
example.describe()

# =============================================================================
# Here 
# =============================================================================

data1 = pd.read_csv(r"E:\Project\new_bike_data.csv")
data1.info()

#Analysing "km_driven" column
data1.isna().sum()
data1["km_driven"].dtypes
np.unique(data1.km_driven)
data1.km_driven = np.where(data1.km_driven.str.contains("Mileage"),np.nan,data1["km_driven"])
data1.km_driven = np.where(data1.km_driven.str.contains("Yes  "),np.nan,data1.km_driven)
data1.isna().sum()

data1.dropna(subset = ["km_driven"],inplace = True)
data1.km_driven.describe()

data1["km_driven"] = data1["km_driven"].astype("float64")
data1["km_driven"].dtypes
data1.info()


data1.to_csv(r"E:\Project\70_percent.csv")
# =============================================================================
# Analysing the "power" column
# =============================================================================

data2 = pd.read_csv(r'E:\Project\70_percent.csv',index_col = 0)
data2.info()
# Filling the brand column using bike_name column
data2['brand'].fillna(data2["bike_name"].str.extract(r'(\b\w+\b)',expand = False),inplace = True)
data2.info()

#data2.isnull().sum()
#data2.brand.mode()
data2.brand.value_counts().index[0]

data2['owner'] = data2.owner.fillna(data2.owner.value_counts()[0])
data2.info()

data2.city.value_counts()
data2['city'].value_counts().index[0]
data2['city'].fillna(data2.city.value_counts()[0],inplace = True)

data2.to_csv(r'E:\Project\completed_data.csv')




data3 = pd.read_csv(r"E:\Project\completed_data.csv",index_col=0)

data4 = data3.copy()

data4.describe()
pd.set_option("display.float_format",lambda x: '%.3f' % x)
data4.describe()
data4.info()
# =============================================================================
# Histogram
# =============================================================================
plt.hist(data4['price'],bins = 5)

# =============================================================================
# Scatter plot
# =============================================================================
plt.scatter(data3['city'],data3['price'],c = 'red',label = "data points")
plt.title("Scatter plot")
plt.xlabel("cities")
plt.ylabel("price")
plt.legend()

data3.describe()
data4.shape[0]
# =============================================================================
# Box plot
# =============================================================================

sns.boxplot(y = data4.price)
sns.boxplot(x = data4.km_driven)

# =============================================================================
# Removing the outliers in numerical variables
# =============================================================================

def get_iqr_values(df, column_name):
    median = df[column_name].mean()
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    max_quantile = q3 + (1.5 * iqr)
    min_quantile = q1 - (1.5 * iqr)
    return median, q1, q3, max_quantile, min_quantile

def remove_outliers(df, column):
    _, _, _, maximum, minimum = get_iqr_values(df, column)
    df_out = df[(df[column] > minimum) & (df[column] < maximum)]
    return df_out

df_out = remove_outliers(data4, 'price')
number_outliers = data4.shape[0] - df_out.shape[0]
print('Number of outliers:', number_outliers)

sns.boxplot(x = df_out.price)

df_out = remove_outliers(data4,'km_driven')
number_outliers = data4.shape[0] - df_out.shape[0]
print("number of outliers:",number_outliers)

sns.boxplot(df_out.km_driven)

df_out.describe()
df_out.info()

data5 = df_out.copy()
# =============================================================================
# Processing Categorial Variables
# =============================================================================

pd.crosstab(index = data5["owner"],columns = "count")
data5.owner = data5["owner"].str.title()

pd.crosstab(index = data5["owner"],columns = "count")

data5 = data5[data5['owner'] != '60206']

pd.crosstab(index = data5["owner"],columns = "count")



data5.describe()
data5.info()





sns.distplot(data5['price'])
# Checking the skewness of price - It is +ve skew
data5.price.skew()

sns.distplot(data5["km_driven"],bins = 7)
data5.km_driven.skew()


sns.countplot(x = 'owner',data = data5)
sns.countplot(x = 'brand',data = data5)

# =============================================================================
# Relation between variables
# =============================================================================

pd.crosstab(index = data5['city'],columns = data5['brand'])
pd.crosstab(index = data5['city'],columns = data5.km_driven)
pd.crosstab(index = data5["owner"],columns = data5.city)

data5.to_csv(r"E:\Project\29-09-2023.csv")


# =============================================================================
# Correlation
# =============================================================================
# Reading the data 
data6 = pd.read_csv(r"E:\Project\29-09-2023.csv",index_col=0)

# data summary
data6.info()

# Copying data to another dataframe
data7 = data6.copy()


# correlating columns of numerical data
numerical_data = data7.select_dtypes(exclude = [object])
corr_matrix = numerical_data.corr()
print(corr_matrix)


data7.describe()
# Setting numerical data to 3 decimal values
pd.set_option("display.float_format",lambda x:"%.3f" % x)
# Checking the datatype of price column
data7.price.dtypes
# Converting int64 to float64
data7.price = data7.price.astype("float64")
# Checking datatype of km_driven
data7.km_driven.dtypes
# Converting int64 to float64
data7.km_driven = data7.km_driven.astype("float64")
data7.info()

# Checking joint probability
pd.crosstab(index = data7.price,columns = data7.km_driven,normalize = True)
sns.boxplot(x = data7.price)
sns.boxplot(x = data7.km_driven)


def get_iqr_values(df, column_name):
    median = df[column_name].mean()
    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1
    max_quantile = q3 + (1.5 * iqr)
    min_quantile = q1 - (1.5 * iqr)
    return median, q1, q3, max_quantile, min_quantile

def remove_outliers(df, column):
    _, _, _, maximum, minimum = get_iqr_values(df, column)
    df_out = df[(df[column] > minimum) & (df[column] < maximum)]
    return df_out

df_out = remove_outliers(data7, 'price')
number_outliers = data7.shape[0] - df_out.shape[0]
print('Number of outliers:', number_outliers)

sns.boxplot(x = df_out.price)
df_out.info()
df_out.describe()
sns.boxplot(x = df_out.km_driven)
pd.crosstab(index = data7.price,columns = data7.km_driven,normalize = True)

numerical_data = df_out.select_dtypes(exclude = [object])
numerical_data.corr()

pd.crosstab(index = df_out.brand,columns = "count" )
sns.countplot(y = df_out.brand)
sns.countplot(x = df_out.brand,hue = df_out.owner)

df_out.info()

df_out.to_csv(r"E:\Project\30-09-2023.csv")





df_out = pd.read_csv(r"E:\Project\30-09-2023.csv")

# =============================================================================
# Model Building
# =============================================================================

# Importing libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,explained_variance_score


# Converting categorical variables into dummy variables


df_out = pd.get_dummies(df_out,drop_first=True)


# Seperating the input and output features

X = df_out.drop(['price'],axis = "columns",inplace = False)
y = df_out.price
# =============================================================================
prices = pd.DataFrame({"1.Before":y,"2.After":np.log(y)})
prices.hist()
# =============================================================================

# Splitting the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# =============================================================================
# Linear Regression
# =============================================================================
# create linear regression model
line_model = LinearRegression()

# Train the model by Fitting the model with training data
line_train_model = line_model.fit(X_train,y_train)

# Perform Predictions using test data
line_predictions = line_model.predict(X_test)

mse = mean_squared_error(y_test,line_predictions)
print(mse)
rmse = mean_squared_error(y_test, line_predictions,squared = False)
print(rmse)
r2 = r2_score(y_test,line_predictions)
print("r-squared value:",r2)
# =============================================================================
# Decision Tree Regressor
# =============================================================================

# create decision tree regressor model
decision_model = DecisionTreeRegressor()

# Train the model by fitting model with training data
decision_train_model = decision_model.fit(X_train,y_train)

# Perform predictions using test data
decision_predictions = decision_model.predict(X_test)

# mean squared error - metric
mse = mean_squared_error(y_test, decision_predictions)
print(mse)

# root mean squared error - metric
rmse = mean_squared_error(y_test, decision_predictions,squared = False)
print(rmse)

# r2_score - metric
decision_tree_r2 = r2_score(y_test,decision_predictions)
print(decision_tree_r2)


# =============================================================================
# Random Forest Regressor
# =============================================================================

# Create Random Forest regressor model
forest_model = RandomForestRegressor()

# Train the model by fitting the model with training data
forest_train_model = forest_model.fit(X_train,y_train)

# Perform predictions using test data
forest_predictions = forest_train_model.predict(X_test)
