

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier, BaggingClassifier, StackingClassifier, GradientBoostingClassifier
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
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import apriori, association_rules




df = pd.read_csv("Airbnb_open_data.csv",low_memory=False)
df.head(5)



df.columns = [col.lower().replace(" ","_") for col in df.columns]
df.head(3)


df.columns


df.shape



df['price'] = df['price'].str.replace('$', '', regex=False).str.replace(',', '').astype(float)



mean_price = df.groupby('neighbourhood_group')['price'].mean()
mean_price = df['price'].mean()
df['price'] = df['price'].fillna(mean_price)
print(mean_price)


df['price'].describe()



plt.figure(figsize=(8, 6))
plt.boxplot(df['price'], vert=False)
plt.title('Box Plot for price')
plt.xlabel('price')
plt.show()


threshold_price = 625
df['price_category'] = df['price'].apply(lambda x: 'Expensive' if x > threshold_price else 'Standard')
df = df.drop(columns=['price'])
print(df.head())

selected_features = ['price_category','instant_bookable']
df_association = df[selected_features].dropna()
for column in df_association.select_dtypes(include=['object']).columns:
    df_association[column] = df_association[column].astype(str)

for column in df_association.columns:
    unique_values = df_association[column].unique()
    print(f"{column}: {unique_values}")
df_association_onehot = pd.get_dummies(df_association)
frequent_itemsets = apriori(df_association_onehot, min_support=0.06, use_colnames=True)
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.01)
print(rules)

rules




