#------------------------------- 5805 ML Term Project -------------------------------------
#------------------------------- Atharva Chouthai ----------------------------------------
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import scipy.stats as stats
from sklearn.decomposition import PCA
from numpy.linalg import cond, svd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',100)


df = pd.read_csv("Airbnb_Open_Data.csv",low_memory=False)
df.head(5)


# In[ ]:

print("----------------------------------------------------------------------------------")
print("                             Project Phase 1 - Regression                            ")
print("----------------------------------------------------------------------------------")

df.columns = [col.lower().replace(" ","_") for col in df.columns]
df.head(3)

print("------------Exploring the dataset ---------------")

print("Columns of dataset:",df.columns)
print("Shape of dataset",df.shape)
print("information about dataset",df.info)
print("Distribustion of dataset",df.describe())

print("------------Exploring the target variable ---------------")

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Distribution of Price')
plt.show()

print("Price check",df['price'].head())
df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '').astype(float)
mean_price = df.groupby('neighbourhood_group')['price'].mean()
mean_price = df['price'].mean()
df['price'] = df['price'].fillna(mean_price)
print(mean_price)
df['price'].describe()

print("------------Exploring and cleaning service fee feature ---------------")

df['service_fee'] = df['service_fee'].str.replace('$', '', regex=False).str.replace(',', '').astype(float)
mean_price = df.groupby('neighbourhood_group')['service_fee'].mean()
mean_price = df['service_fee'].mean()
df['service_fee'] = df['service_fee'].fillna(mean_price)
print(mean_price)

df['service_fee'].describe()

plt.figure(figsize=(8, 6))
plt.boxplot(df['service_fee'], vert=False)
plt.title('Box Plot for service_fee')
plt.xlabel('service_fee')
plt.show()

df['service_fee'] = df['service_fee'].clip(upper=100)
df['service_fee'] = df['service_fee'].clip(lower=40)
Q1 = df['service_fee'].quantile(0.25)
Q3 = df['service_fee'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['service_fee'] < lower_bound) | (df['service_fee'] > upper_bound)]
df = df[(df['service_fee'] >= lower_bound) & (df['service_fee'] <= upper_bound)]
df.isnull().sum()

print("------------Exploring and cleaning available 365 feature ---------------")

mean_price = df.groupby('neighbourhood_group')['availability_365'].mean()
mean_price = df['availability_365'].mean()
df['availability_365'] = df['availability_365'].fillna(mean_price)
print(mean_price)
df = df[df['availability_365'] >0]
df['availability_365'] = df['availability_365'].clip(upper=365)
df['availability_365'].describe()
Q1 = df['availability_365'].quantile(0.25)
Q3 = df['availability_365'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['availability_365'] < lower_bound) | (df['availability_365'] > upper_bound)]
df = df[(df['availability_365'] >= lower_bound) & (df['availability_365'] <= upper_bound)]
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['availability_365'])
plt.title('Box Plot Before Outlier Removal')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['availability_365'])
plt.title('Box Plot After Outlier Removal')
plt.show()
df['availability_365'].describe()

print("------------Exploring and cleaning host listings feature ---------------")

mean_price = df.groupby('neighbourhood_group')['calculated_host_listings_count'].mean()
mean_price = df['calculated_host_listings_count'].mean()
df['calculated_host_listings_count'] = df['calculated_host_listings_count'].fillna(mean_price)
print(mean_price)
df['calculated_host_listings_count'].describe()
plt.figure(figsize=(8, 6))
plt.boxplot(df['calculated_host_listings_count'], vert=False)
plt.title('Box Plot for Host listing')
plt.xlabel('Host Listings')
plt.show()
Q1 = df['calculated_host_listings_count'].quantile(0.25)
Q3 = df['calculated_host_listings_count'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['calculated_host_listings_count'] < lower_bound) | (df['calculated_host_listings_count'] > upper_bound)]
df = df[(df['calculated_host_listings_count'] >= lower_bound) & (df['calculated_host_listings_count'] <= upper_bound)]
plt.figure(figsize=(8, 6))
plt.boxplot(df['calculated_host_listings_count'], vert=False)
plt.title('Box Plot for Host listing')
plt.xlabel('Host Listings')
plt.show()

df['calculated_host_listings_count'].describe()

print("------------Exploring and cleaning minimum nights feature ---------------")

mean_price = df.groupby('neighbourhood_group')['minimum_nights'].mean()
mean_price = df['minimum_nights'].mean()
df['minimum_nights'] = df['minimum_nights'].fillna(mean_price)
print(mean_price)
df = df[df['minimum_nights'] >0]
df['minimum_nights'].describe()
Q1 = df['minimum_nights'].quantile(0.25)
Q3 = df['minimum_nights'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['minimum_nights'] < lower_bound) | (df['minimum_nights'] > upper_bound)]
df = df[(df['minimum_nights'] >= lower_bound) & (df['minimum_nights'] <= upper_bound)]
df['minimum_nights'].describe()

print("------------Exploring and cleaning Host identity feature ---------------")
print("Host_identity:",df['host_identity_verified'].head())
df['host_identity_verified'].replace({True:1,False:0},inplace=True)
df['host_identity_verified'].describe()

print("------------Exploring and cleaning Reviews per month feature ---------------")
df["reviews_per_month"].fillna(df["reviews_per_month"].mean(),inplace=True)
plt.figure(figsize=(8, 6))
plt.boxplot(df['reviews_per_month'], vert=False)
plt.title('Box Plot for Host listing')
plt.xlabel('Host Listings')
plt.show()
df['reviews_per_month'].describe()
Q1 = df['reviews_per_month'].quantile(0.25)
Q3 = df['reviews_per_month'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['reviews_per_month'] < lower_bound) | (df['reviews_per_month'] > upper_bound)]
df = df[(df['reviews_per_month'] >= lower_bound) & (df['reviews_per_month'] <= upper_bound)]
plt.figure(figsize=(8, 6))
plt.boxplot(df['reviews_per_month'], vert=False)
plt.title('Box Plot for Host listing')
plt.xlabel('Host Listings')
plt.show()
df['reviews_per_month'].describe()

print("------------Exploring and cleaning Neighbourhood group feature ---------------")

df["neighbourhood_group"].fillna(df["neighbourhood_group"].mode(),inplace=True)
df["neighbourhood_group"].describe()

print("------------Exploring and cleaning Neighbourhood feature ---------------")
df["neighbourhood"].fillna(df["neighbourhood"].mode(),inplace=True)

print("------------Exploring and cleaning cancellation policy feature ---------------")
df['cancellation_policy'].replace({"moderate":0,"flexible":1,"strict":2},inplace=True)
df['cancellation_policy'].describe()

print("------------Exploring and cleaning Number of reviews feature ---------------")
df["number_of_reviews"].fillna(df["number_of_reviews"].mean(),inplace=True)
plt.figure(figsize=(8, 6))
plt.boxplot(df['number_of_reviews'], vert=False)
plt.title('Box Plot for number_of_reviews')
plt.xlabel('number_of_reviews')
plt.show()
df['number_of_reviews'].describe()
Q1 = df['number_of_reviews'].quantile(0.25)
Q3 = df['number_of_reviews'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['number_of_reviews'] < lower_bound) | (df['number_of_reviews'] > upper_bound)]
df = df[(df['number_of_reviews'] >= lower_bound) & (df['number_of_reviews'] <= upper_bound)]
plt.figure(figsize=(8, 6))
plt.boxplot(df['number_of_reviews'], vert=False)
plt.title('Box Plot for number_of_reviews')
plt.xlabel('number_of_reviews')
plt.show()
df['number_of_reviews'].describe()

print("------------Exploring and cleaning Construction year feature ---------------")
df["age_property"] = 2023 - df["construction_year"] 
df.drop(columns = ["construction_year"], inplace = True)
df["age_property"].fillna(df["age_property"].mean(),inplace=True)
plt.figure(figsize=(8, 6))
plt.boxplot(df['age_property'], vert=False)
plt.title('Box Plot for age_property')
plt.xlabel('age_property')
plt.show()
df['age_property'].describe()

print("------------Exploring and cleaning instant bookable feature ---------------")

df['instant_bookable'].replace({True:1,False:0},inplace=True)

df.dropna(subset=['neighbourhood_group'], inplace=True)
df.dropna(subset=['neighbourhood'], inplace=True)
df.dropna(subset=['instant_bookable'], inplace=True)
df.dropna(subset=['cancellation_policy'], inplace=True)
df.dropna(subset=['lat'], inplace=True)
df.dropna(subset=['long'], inplace=True)

df_num = df.select_dtypes(include=np.number)
print(df_num)

print("------------ Standardizing the required features ---------------")

def standardize(data):
  mean = np.mean(data)
  std = np.std(data)

  standardized_data = (data - mean) / std
  return standardized_data


df['service_fee'] = standardize(df['service_fee'])
df['host_id'] = standardize(df['host_id'])
df['minimum_nights'] = standardize(df['minimum_nights'])
df['number_of_reviews'] = standardize(df['number_of_reviews'])
df['reviews_per_month'] = standardize(df['reviews_per_month'])
df['calculated_host_listings_count'] = standardize(df['calculated_host_listings_count'])
df['availability_365'] = standardize(df['availability_365'])
df['age_property'] = standardize(df['age_property'])

print("------------ Covariance Matrix ---------------")

covariance_matrix = df.cov()
plt.figure(figsize=(20, 8))
sns.heatmap(covariance_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Sample Covariance Matrix')
plt.show()
print(covariance_matrix)


print("------------ Correlation Matrix ---------------")


correlation_matrix = df.corr()
plt.figure(figsize=(20, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", linewidths=.5)
plt.title('Sample Correlation Matrix')
plt.show()
print(correlation_matrix)

unwanted_cols = ['host_id','name','lat','long','review_rate_number','host_name','last_review','country','country_code','id','house_rules','license','neighbourhood']
df.drop(unwanted_cols, axis = 1, inplace=True)

print("------------ Encoding Categorical Features ---------------")

categorical_data = df.select_dtypes(exclude=np.number).columns
print(categorical_data)

encoded_categorical_df = pd.get_dummies(df[categorical_data], drop_first=True)
df.drop(['neighbourhood_group', 'room_type','host_identity_verified'],axis=1,inplace=True)
df = pd.concat([df,encoded_categorical_df],axis=1)
print(df.head())
print("Shape of data after encoding ",df.shape)


print("------------ Splitting into X and Y ---------------")

y = df['price']
X = df.drop('price', axis=1)
print("y", y.head())
print("X", X.head())

print("------------ Random Forest Analysis for Regression  ---------------")

rf_model = RandomForestRegressor(n_estimators = 100,
    max_features = 8,
    random_state=5805)
rf_model.fit(X, y)

feature_importance = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)
print(feature_importance_df.head())

N = 10  # Adjust N as needed
top_features = feature_importance_df.head(N)
print(top_features)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_features['Importance'], y=top_features['Feature'], palette='viridis')
plt.title('Top 10 Features by Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

threshold = 0.005

selected_features_df = feature_importance_df[feature_importance_df["Importance"] >= threshold]
eliminated_features_df = feature_importance_df[feature_importance_df["Importance"] < threshold]

print("Selected Features:")
print(selected_features_df)

selected_features = selected_features_df['Feature'].values
df_selected = df[selected_features]

print("------------ PCA for Regression  ---------------")

pca = PCA(random_state= 5805)
X_pca = pca.fit_transform(X)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
num_components_needed = np.argmax(cumulative_explained_variance > 0.90) + 1
print(f"Number of principal components needed to explain more than 90% of the variance: {num_components_needed}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o', linestyle='-', color='b')
plt.title('Cumulative Explained Variance vs. Number of Components (PCA) ')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
plt.axvline(x=num_components_needed, color='g', linestyle='--', label=f'{num_components_needed} Components')
plt.legend()
plt.grid(True)
plt.show()

cov_matrix = pca.get_covariance()
condition_number = cond(cov_matrix)
print(f"Condition number before passing n_components:{condition_number}")

pca_2 = PCA(n_components=num_components_needed)
X_pca = pca_2.fit_transform(X)

cov_matrix = pca_2.get_covariance()
condition_number = cond(cov_matrix)
print(f"Condition number after passing n_components :{condition_number}")

print("------------ SVD for Regression  ---------------")
from sklearn.decomposition import TruncatedSVD

explained_variance_ratio = []
for n in range(1, X.shape[1]):
    svd = TruncatedSVD(n_components=n)
    svd.fit(X)
    explained_variance_ratio.append(svd.explained_variance_ratio_.sum())
plt.figure(figsize=(10, 6))
plt.plot(range(1, X.shape[1]), explained_variance_ratio, marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratio vs. Number of Components (TruncatedSVD)')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
plt.axvline(x=num_components_needed, color='g', linestyle='--', label=f'{num_components_needed} Components')
plt.legend()
plt.grid(True)
plt.show()

print("------------ VIF for Regression  ---------------")

vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif.replace([np.inf, -np.inf], np.nan, inplace=True)
vif.dropna(inplace=True)
# vif
vif_sorted = vif.sort_values(by='VIF', ascending=False)
print(vif_sorted)

print("No.of Vifs :",len(vif))

print("----------------------------------------------------------------------------------")
print("                             Project Phase 2 - Regression                            ")
print("----------------------------------------------------------------------------------")

print("------------ Splitting the dataset into training and testing for Regression  ---------------")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)
print("X test shape ", X_test.shape)
print("X train:",X_train.head())

print("------------ Applying Linear Regression  ---------------")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2_train = round(r2_score(y_train, y_pred_train),2)
r2_test = round(r2_score(y_test, y_pred_test),2)

print(f'R-squared (R2) on training set: {r2_train}')
print(f'R-squared (R2) on testing set: {r2_test}')

mse_train = round(mean_squared_error(y_train, y_pred_train),2)
mse_test = round(mean_squared_error(y_test, y_pred_test),2)

print(f'Mean Squared Error (MSE) on training set: {mse_train}')
print(f'Mean Squared Error (MSE) on testing set: {mse_test}')

rmse_train = round(mean_squared_error(y_train, y_pred_train, squared=False),2)
rmse_test = round(mean_squared_error(y_test, y_pred_test, squared=False),2)

print(f'Root Mean Squared Error (RMSE) on training set: {rmse_train}')
print(f'Root Mean Squared Error (RMSE) on testing set: {rmse_test}')

print("------------ Applying Stepwise Regression (OLS)  ---------------")

import pandas as pd
import statsmodels.api as sm

trainCols = list(X_train.columns)
featuresDropped = []
threshold = 0.01

resultsTable = pd.DataFrame(columns=['Feature Dropped', 'AIC', 'BIC', 'AdjR2', 'P-Value'])
while len(trainCols) > 0:
    X_train_stepwise = X_train[trainCols]
    X_train_stepwise = sm.add_constant(X_train_stepwise)
    model = sm.OLS(y_train, X_train_stepwise).fit()
    pValues = model.pvalues[1:]
    maxPVal = pValues.max()
    maxPValIdx = pValues.idxmax()

    if maxPVal > threshold:
        dropFeature = maxPValIdx
        featuresDropped.append(dropFeature)
        trainCols.remove(dropFeature)
        print('Feature Dropped:', dropFeature)
        print(model.summary())

        f_statistic = model.fvalue.round(2)
        f_p_value = model.f_pvalue
        aic = model.aic.round(2)
        bic = model.bic.round(2)
        adj_r_squared = model.rsquared_adj.round(2)
        confidence_intervals = round(model.conf_int()), 2

        resultsTable = resultsTable.append({
            'Feature Dropped': dropFeature,
            'AIC': aic,
            'BIC': bic,
            'AdjR2': adj_r_squared,
            'P-Value': maxPVal,
            'F-Value': f_statistic
        }, ignore_index=True)

        print(f"F-statistic: {f_statistic}")
        print(f"P-value (F-test): {f_p_value}")
        print(f"AIC: {aic}")
        print(f"BIC: {bic}")
        print(f"Adjusted R-squared: {adj_r_squared}")
        print("Confidence Intervals:")
        print(confidence_intervals)

    else:
        break

print("\nResults Table:")
print(resultsTable)

print("-------------------- Prediction Intervals plotting ---------------------")
print("X test shape ", X_test.shape)
model1 = sm.OLS(y_train, X_train).fit()

sm_pred = model1.get_prediction((X_test)).summary_frame(alpha=0.05)
lower_interval = sm_pred['obs_ci_lower']
upper_interval = sm_pred['obs_ci_upper']
# y_test_original = reverse_standardized_data(sm_pred['mean'], y_original)
# lower_interval_original = reverse_standardized_data(lower_interval, y_original)
# upper_interval_original = reverse_standardized_data(upper_interval, y_original)
# print('Upper Interval:',upper_interval_original.head())
x_range = [i for i in range(len(y_test))]
plt.plot(x_range, lower_interval, alpha=0.4, label='Lower interval')
plt.plot(x_range, upper_interval, alpha=0.4, label='Upper interval')
plt.plot(x_range, y_test, alpha=1.0, label='predicted Values')
plt.title('Predicted Values with Intervals')
plt.ylabel('Price')
plt.xlabel('No.of samples')
plt.legend()
plt.show()

print("------------ Applying Linear Regression after OLS  ---------------")

model = LinearRegression()
model.fit(X_train[trainCols], y_train)

y_pred_train = model.predict(X_train[trainCols])
y_pred_test = model.predict(X_test[trainCols])

r2_train = round(r2_score(y_train, y_pred_train),2)
r2_test = round(r2_score(y_test, y_pred_test),2)

print(f'R-squared (R2) on training set: {r2_train}')
print(f'R-squared (R2) on testing set: {r2_test}')

mse_train = round(mean_squared_error(y_train, y_pred_train),2)
mse_test = round(mean_squared_error(y_test, y_pred_test),2)

print(f'Mean Squared Error (MSE) on training set: {mse_train}')
print(f'Mean Squared Error (MSE) on testing set: {mse_test}')

rmse_train = round(mean_squared_error(y_train, y_pred_train, squared=False),2)
rmse_test = round(mean_squared_error(y_test, y_pred_test, squared=False),2)

print(f'Root Mean Squared Error (RMSE) on training set: {rmse_train}')
print(f'Root Mean Squared Error (RMSE) on testing set: {rmse_test}')


train_range = [i for i in range(len(y_train))]
test_range = [i for i in range(y_test.shape[0])]
plt.plot(train_range, y_train, label='Training Price')
plt.plot(test_range, y_test, label='Test Price')
plt.plot(test_range, y_pred_test, label='Predicted Price')
plt.xlabel('Observations')
plt.ylabel('Price')
plt.legend(loc='best')
plt.title('Linear Regression (OLS Analysis)')
plt.show()

from prettytable import PrettyTable

newTable = PrettyTable(["Regression Model", "R2", "Adjusted R2", "AIC", "BIC", "MSE", "RMSE"])

newTable.add_row(["Linear Regression", r2_test, adj_r_squared, aic, bic, mse_test, rmse_test])

print(newTable)


