import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,cross_val_score,StratifiedKFold
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
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix, recall_score, roc_auc_score, roc_curve,classification_report,roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("Airbnb_Open_Data.csv", low_memory=False)
df.head(5)

print("------------------------------------------------------------")
print("                      Data Cleaning                         ")
print("------------------------------------------------------------")

df.columns = [col.lower().replace(" " , "_") for col in df.columns]
print(df.head(3))

print(df.columns)

print(df.shape)


print("         1. Cleaning Price feature           ")

print("a. Removing $ sign.")
print("b. Filling the missing values with neighbourhood.")


df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '').astype(float)


mean_price = df.groupby('neighbourhood_group')['price'].mean()
mean_price = df['price'].mean()
df['price'] = df['price'].fillna(mean_price)
print(mean_price)


print("         2. Cleaning Service Fee feature         ")
print("\n a. Removing $ sign.")
print("\n b. Filling the missing values with neighbourhood.")

df['service_fee'] = df['service_fee'].str.replace('$', '', regex=False).str.replace(',', '').astype(float)

mean_price = df.groupby('neighbourhood_group')['service_fee'].mean()
mean_price = df['service_fee'].mean()
df['service_fee'] = df['service_fee'].fillna(mean_price)
print(mean_price)


print("         3. Cleaning availability (365) feature.           ")
print("\n a. Filling the missing values with neighbourhood.")

mean_availability = df.groupby('neighbourhood_group')['availability_365'].mean()
mean_availability = df['availability_365'].mean()
df['availability_365'] = df['availability_365'].fillna(mean_price)
print(mean_availability)


plt.figure(figsize=(8, 6))
plt.boxplot(df['availability_365'], vert=False)
plt.title('Box Plot for availability_365')
plt.xlabel('availability_365')
plt.show()

df = df[df['availability_365'] >0]
df['availability_365'] = df['availability_365'].clip(upper=365)
df['availability_365'].describe()


print("         4. Cleaning Host Listings feature           ")
print("\n a. Filling the missing values using the neighbourhood.")


mean_hostings = df.groupby('neighbourhood_group')['calculated_host_listings_count'].mean()
mean_hostings = df['calculated_host_listings_count'].mean()
df['calculated_host_listings_count'] = df['calculated_host_listings_count'].fillna(mean_price)
print(mean_hostings)


print("         5. Cleaning Minimum Nights feature          ")
print("\n a. Filling the missing values using the neighbourhood.")

mean_minnights = df.groupby('neighbourhood_group')['minimum_nights'].mean()
mean_minnights = df['minimum_nights'].mean()
df['minimum_nights'] = df['minimum_nights'].fillna(mean_price)
print(mean_minnights)

df.isnull().sum()

df = df[df['minimum_nights'] >0]
df['minimum_nights'].describe()

plt.figure(figsize=(8, 6))
plt.boxplot(df['minimum_nights'], vert=False)
plt.title('Box Plot for minimum_nights')
plt.xlabel('Minimum Nights')
plt.show()


Q1 = df['minimum_nights'].quantile(0.25)
Q3 = df['minimum_nights'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Plot a box plot before and after outlier removal
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['minimum_nights'])
plt.title('Box Plot Before Outlier Removal')

# Identify and remove outliers
outliers = df[(df['minimum_nights'] < lower_bound) | (df['minimum_nights'] > upper_bound)]
df = df[(df['minimum_nights'] >= lower_bound) & (df['minimum_nights'] <= upper_bound)]

plt.subplot(1, 2, 2)
sns.boxplot(x=df['minimum_nights'])
plt.title('Box Plot After Outlier Removal')

plt.show()
df['minimum_nights'].describe()


print("         5. Transforming host identity feature           ")
df['host_identity_verified'].replace({True:1,False:0},inplace=True)


print("         6. Cleaning Reviews per Month            ")
df["reviews_per_month"].fillna(df["reviews_per_month"].mean(),inplace=True)


print("         7. Cleaning Neighbourhood Group and Neighbourhoods feature          ")
df["neighbourhood_group"].fillna(df["neighbourhood_group"].mode(),inplace=True)
df["neighbourhood"].fillna(df["neighbourhood"].mode(),inplace=True)


print("         8. Transforming Cancellation Policy feature         ")
df['cancellation_policy'].replace({"moderate":0,"flexible":1,"strict":2},inplace=True)


print("         9. Cleaning Number of reviews           ")
df["number_of_reviews"].fillna(df["number_of_reviews"].mean(),inplace=True)
df.isnull().sum()


print("         10. Cleaning Construction year feature          ")
df["age_property"] = 2023 - df["construction_year"]
df.drop(columns = ["construction_year"], inplace = True)
df["age_property"].fillna(df["age_property"].mean(),inplace=True)


print("         11. Transforming instant bookable feature           ")
df['instant_bookable'].replace({True:1,False:0},inplace=True)

df.dropna(subset=['neighbourhood_group'], inplace=True)
df.dropna(subset=['neighbourhood'], inplace=True)
df.dropna(subset=['instant_bookable'], inplace=True)
df.dropna(subset=['cancellation_policy'], inplace=True)

df.head()
df.shape


print(" 12. Removing unwanted columns")

unwanted_cols = ['lat','long','name','host_id','review_rate_number','last_review','country','country_code','host_name','id','house_rules','license','service_fee']
df.drop(unwanted_cols, axis = 1, inplace=True)


print("------------------------------------------------------------")
print("                      Encoding Categorical variables                         ")
print("------------------------------------------------------------")


categorical_data = df.select_dtypes(exclude=np.number).columns
print(categorical_data)

encoded_categorical_df = pd.get_dummies(df[categorical_data], drop_first=True)
df.drop(['neighbourhood_group', 'room_type','host_identity_verified', 'neighbourhood'],axis=1,inplace=True)
# df.drop(['host_identity_verified', 'neighbourhood'],axis=1,inplace=True)
df = pd.concat([df,encoded_categorical_df],axis=1)

df.head()


print("------------------------------------------------------------")
print("                      Standardize necessary features                         ")
print("------------------------------------------------------------")

def standardize(data):
  mean = np.mean(data)
  std = np.std(data)

  standardized_data = (data - mean) / std
  return standardized_data


df['minimum_nights'] = standardize(df['minimum_nights'])
df['number_of_reviews'] = standardize(df['number_of_reviews'])
df['reviews_per_month'] = standardize(df['reviews_per_month'])
df['calculated_host_listings_count'] = standardize(df['calculated_host_listings_count'])
df['availability_365'] = standardize(df['availability_365'])


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


booking_features = ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count', 'availability_365']

df_booking = df.dropna(subset=booking_features)

X_booking = df_booking[booking_features]


k_values = range(2, 11)

silhouette_scores = []
wcss_values = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=5, random_state=5805)
    labels = kmeans.fit_predict(X_booking)

    silhouette_avg = silhouette_score(X_booking, labels)
    silhouette_scores.append(silhouette_avg)

    wcss_values.append(kmeans.inertia_)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.grid(True)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Analysis')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')

plt.subplot(1, 2, 2)
plt.plot(k_values, wcss_values, marker='o')
plt.title('Within-Cluster Variation (WCSS) Analysis')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')

plt.tight_layout()
plt.grid(True)
plt.show()

