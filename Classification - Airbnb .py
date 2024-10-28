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


# ## Engineering Host type 

minimum_reviews_per_month = 1.32
minimum_listings_count = 1
minimum_availability_365 = 30

df['Host_type'] = 'normal host'
mask = (
    (df['reviews_per_month'] >= minimum_reviews_per_month) &
    (df['calculated_host_listings_count'] >= minimum_listings_count) &
    (df['availability_365'] >= minimum_availability_365)
)
df.loc[mask, 'Host_type'] = 'superhost'

print(df['Host_type'].value_counts())

sns.countplot(x='Host_type',data=df)
df.replace({'normal host':0, 'superhost':1},inplace=True)
df.drop(['reviews_per_month'],axis=1,inplace=True)

print("------------------------------------------------------------")
print("                      Standardize necessary features                         ")
print("------------------------------------------------------------")

def standardize(data):
  mean = np.mean(data)
  std = np.std(data)

  standardized_data = (data - mean) / std
  return standardized_data

df['price'] = standardize(df['price'])
df['minimum_nights'] = standardize(df['minimum_nights'])
df['number_of_reviews'] = standardize(df['number_of_reviews'])
df['calculated_host_listings_count'] = standardize(df['calculated_host_listings_count'])
df['availability_365'] = standardize(df['availability_365'])
df['age_property'] = standardize(df['age_property'])

df.head()

print("------------------------------------------------------------")
print("                      Splitting Dataset into independent and dependent variables                         ")
print("------------------------------------------------------------")

y = df['Host_type']
X = df.drop('Host_type', axis=1)

print(X.head())

print(y.shape)
print(X.shape)

print("------------------------------------------------------------")
print("                      Feature importance using Random Forest                         ")
print("------------------------------------------------------------")

rf_model = RandomForestClassifier(n_estimators = 200,
    random_state=5805)
rf_model.fit(X, y)

feature_importance = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": X.columns, "Importance": feature_importance})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

feature_importance_df.head()

N = 100  # Adjust N as needed
top_features = feature_importance_df.head(N)
print(top_features)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_features['Importance'], y=top_features['Feature'], palette='viridis')
plt.title('Top Features by Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

threshold = 0.01

# Filter features based on the threshold
selected_features_df = feature_importance_df[feature_importance_df["Importance"] >= threshold]
eliminated_features_df = feature_importance_df[feature_importance_df["Importance"] < threshold]

# Display selected and eliminated features
print(f"Selected Features: {selected_features_df}")

selected_features = selected_features_df['Feature'].values
df = df[selected_features]
print(f'dataframe after selected features: {df}')

print("------------------------------------------------------------")
print("                      PCA classification                        ")
print("------------------------------------------------------------")
pca = PCA(random_state= 5805)
X_pca = pca.fit_transform(X)
# print("This is X_PCA >>>>>>>>>>>>>>", X_pca)

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
print(f"Condition number:{condition_number}")

pca_2 = PCA(n_components=num_components_needed)
X_pca = pca_2.fit_transform(X)

cov_matrix = pca_2.get_covariance()
condition_number = cond(cov_matrix)
print(f"Condition number:{condition_number}")

from sklearn.decomposition import TruncatedSVD

explained_variance_ratio = []
for n in range(1, X.shape[1]):
    svd = TruncatedSVD(n_components=n)
    svd.fit(X)
    explained_variance_ratio.append(svd.explained_variance_ratio_.sum())

# Plotting the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, X.shape[1]), explained_variance_ratio, marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratio vs. Number of Components (TruncatedSVD)')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance Ratio')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Explained')
plt.axvline(x=num_components_needed, color='g', linestyle='--', label=f'{num_components_needed} Components')
plt.grid(True)
plt.show()

print("------------------------------------------------------------")
print("                      VIF                        ")
print("------------------------------------------------------------")
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif.replace([np.inf, -np.inf], np.nan, inplace=True)
vif.dropna(inplace=True)

print(vif)

vif_sorted = vif.sort_values(by='VIF', ascending=False)
print(vif_sorted)

X = df

print("------------------------------------------------------------")
print("                      Balancing the dataset using over sampling                         ")
print("------------------------------------------------------------")

from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=5805)
X, y = os.fit_resample(X, y)

y.value_counts()

# from imblearn.under_sampling import RandomUnderSampler
# us = RandomUnderSampler(sampling_strategy= {0:10000,1:10000}, random_state=5805)
# X, y = us.fit_resample(X, y)

print(X.shape)
print(y.shape)

print("------------------------------------------------------------")
print("                      Splitting Dataset into Training and Testing                         ")
print("------------------------------------------------------------")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=5805)

# X_train[:5]
# y_train[:5]

print("------------------------------------------------------------")
print("                      Decision Tree Classifier                         ")
print("------------------------------------------------------------")

dt_classifier = DecisionTreeClassifier(random_state=5805)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)

y_pred_dt_proba = dt_classifier.predict_proba(X_test)[::,-1]

accuracy = round(accuracy_score(y_test, y_pred),2)
confusion_matrix_basic = confusion_matrix(y_test, y_pred)

print("               Performance Matrices Decision Tree                  ")

print(f"Accuracy of Basic Tree: {accuracy}")
print(f"Confusion matrix of Basic Tree: \n{confusion_matrix_basic}")
sns.heatmap(confusion_matrix_basic, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

basic_tree_roc_auc = round(roc_auc_score(y_test, y_pred_dt_proba),2)

print(f'Basic Tree Roc - Auc {basic_tree_roc_auc}')

TP = confusion_matrix_basic[1][1]
TN = confusion_matrix_basic[0][0]
FP = confusion_matrix_basic[0][1]
FN = confusion_matrix_basic[1][0]

precision = round(TP / (TP + FP),2)
recall = round(TP / (TP + FN),2)
f1_score = round(2 * (precision * recall) / (precision + recall),2)
specificity = round(TN / (TN + FP),2)
print(f"Pricision of Basic Tree: {precision}")
print(f"Recall of Basic Tree: {recall}")
print(f"F1 score of Basic Tree: {f1_score}")
print(f"Specificity of Basic Tree: {specificity}")

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_pred_dt_proba)
auc_tree = roc_auc_score(y_test, y_pred_dt_proba)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Decision Tree (AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Basic Decision Tree')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("               Stratified KFold Decision Tree                   ")

n_splits = 3
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(dt_classifier, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')
print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_basic = round(cv_results.mean(),2)
print(f"Mean AUC Score Basic Decision Tree : {St_kfold_basic}")


print("------------------------------------------------------------")
print("                      Pre-Pruned Tree                         ")
print("------------------------------------------------------------")

tuned_params = {
    'max_depth': [5, 8, 12, 16],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [5,7,8,9,10],
    'splitter': ['best', 'random'],
    'criterion': ['gini', 'entropy']

}

dt_classifier_forGridSearch = DecisionTreeClassifier(random_state=5805)
grid_search = GridSearchCV(dt_classifier, param_grid=tuned_params, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_dt_classifier = DecisionTreeClassifier(**best_params, random_state= 5805)
best_dt_classifier.fit(X_train, y_train)
y_pred_pre = best_dt_classifier.predict(X_test)
y_prob_pre_pruned_tree = best_dt_classifier.predict_proba(X_test)[::, -1]


pre_pruned_accuracy = round(accuracy_score(y_test, y_pred_pre),2)
confusion_matrix_pre = confusion_matrix(y_test, y_pred_pre)
recall_pre = (recall_score(y_test, y_pred_pre))
roc_auc_pre = (roc_auc_score(y_test, y_prob_pre_pruned_tree)).round(2)
sns.heatmap(confusion_matrix_pre, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Best Parameters:", best_params)
print("Accuracy (Pre-pruned) tree :", pre_pruned_accuracy)
print(f"Confusion Matrix (Pre-pruned):\n {confusion_matrix_pre}")
print(f"Roc-Auc (Pre-pruned):\n {roc_auc_pre}")


TP = confusion_matrix_pre[1][1]
TN = confusion_matrix_pre[0][0]
FP = confusion_matrix_pre[0][1]
FN = confusion_matrix_pre[1][0]

precision_pre = round(TP / (TP + FP),2)
recall_pre = round(TP / (TP + FN),2)
f1_score_pre = round(2 * (precision_pre * recall_pre) / (precision_pre + recall_pre),2)
specificity_pre = round(TN / (TN + FP),2)
print(f"Pricision of Pre Pruned Tree: {precision_pre}")
print(f"Recall of Pre Pruned Tree: {recall_pre}")
print(f"F1 score of Pre Pruned Tree: {f1_score_pre}")
print(f"Specificity of Pre Pruned Tree: {specificity_pre}")

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_pre_pruned_tree)
auc_tree = roc_auc_score(y_test, y_prob_pre_pruned_tree)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Decision Tree(Pre-pruned)(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Pre-pruned Decision Tree')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

n_splits = 3
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(best_dt_classifier, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')
print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_pre = round(cv_results.mean(),2)
print(f"Mean AUC Score Pre-Pruned Decision Tree : {St_kfold_pre}")

print("------------------------------------------------------------")
print("                      Pruned Tree                         ")
print("------------------------------------------------------------")
print("               Alphas calculation                  ")

path = dt_classifier.cost_complexity_pruning_path(X_train, y_train)
alphas = path['ccp_alphas']
print(alphas)
print(len(alphas))
print(alphas)
# filtered_alphas = [alpha for alpha in alphas if alpha > 1e-10]

# =============================
# Grid search for best alpha
# =============================
accuracy_train, accuracy_test = [], []
for i in alphas[200:400]:
    dt_classifier = DecisionTreeClassifier(ccp_alpha=i, random_state=5805)
    dt_classifier.fit(X_train, y_train)
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(alphas[200:400], accuracy_train[0:200], marker="o", label="train",
        drawstyle="steps-post")
ax.plot(alphas[200:400], accuracy_test[0:200], marker="o", label="test",
        drawstyle="steps-post")
ax.legend()
plt.grid()
plt.tight_layout()
plt.show()


path = dt_classifier.cost_complexity_pruning_path(X_train, y_train)
alphas = path['ccp_alphas']
print(alphas)
print(len(alphas))
print(alphas)
# filtered_alphas = [alpha for alpha in alphas if alpha > 1e-10]

# =============================
# Grid search for best alpha
# =============================
accuracy_train, accuracy_test = [], []
for i in alphas[500:700]:
    dt_classifier = DecisionTreeClassifier(ccp_alpha=i, random_state=5805)
    dt_classifier.fit(X_train, y_train)
    y_train_pred = dt_classifier.predict(X_train)
    y_test_pred = dt_classifier.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(alphas[500:700], accuracy_train[0:200], marker="o", label="train",
        drawstyle="steps-post")
ax.plot(alphas[500:700], accuracy_test[0:200], marker="o", label="test",
        drawstyle="steps-post")
ax.legend()
plt.grid()
plt.tight_layout()
plt.show()
#
#
# path = dt_classifier.cost_complexity_pruning_path(X_train, y_train)
# alphas = path['ccp_alphas']
# print(alphas)
# print(len(alphas))
# print(alphas)
# # filtered_alphas = [alpha for alpha in alphas if alpha > 1e-10]
#
# # =============================
# # Grid search for best alpha
# # =============================
# accuracy_train, accuracy_test = [], []
# for i in alphas[2500:2900]:
#     dt_classifier = DecisionTreeClassifier(ccp_alpha=i, random_state=5805)
#     dt_classifier.fit(X_train, y_train)
#     y_train_pred = dt_classifier.predict(X_train)
#     y_test_pred = dt_classifier.predict(X_test)
#     accuracy_train.append(accuracy_score(y_train, y_train_pred))
#     accuracy_test.append(accuracy_score(y_test, y_test_pred))
# fig, ax = plt.subplots()
# ax.set_xlabel('alpha')
# ax.set_ylabel('accuracy')
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(alphas[2500:2900], accuracy_train[0:400], marker="o", label="train",
#         drawstyle="steps-post")
# ax.plot(alphas[2500:2900], accuracy_test[0:400], marker="o", label="test",
#         drawstyle="steps-post")
# ax.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()


pruned_tree = DecisionTreeClassifier(ccp_alpha=0.0001125, random_state=5805)
pruned_tree.fit(X_train, y_train)
y_pred_pruned = pruned_tree.predict(X_test)
y_prob_pruned_tree = pruned_tree.predict_proba(X_test)[::,-1]


print("               Performance Matrices Decision Tree (Pruned)                  ")

# Calculating metrics for the pruned tree
accuracy_pruned = round(accuracy_score(y_test, y_pred_pruned),2)
confusion_matrix_pruned = confusion_matrix(y_test, y_pred_pruned)
recall_pruned = round(recall_score(y_test, y_pred_pruned),3)
roc_auc_pruned = round(roc_auc_score(y_test, y_prob_pruned_tree),2)
sns.heatmap(confusion_matrix_pruned, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Displaying the results
print("Metrics for Pruned Tree:")
print(f"Accuracy: {accuracy_pruned}")
print("Confusion Matrix:")
print(confusion_matrix_pruned)
# print(f"Recall: {recall_pruned}")
print(f"AUC: {roc_auc_pruned}")

#
TP = confusion_matrix_pruned[1][1]
TN = confusion_matrix_pruned[0][0]
FP = confusion_matrix_pruned[0][1]
FN = confusion_matrix_pruned[1][0]

precision_post = round(TP / (TP + FP),2)
recall_post = round(TP / (TP + FN),2)
f1_score_post = round(2 * (precision_post * recall_post) / (precision_post + recall_post),2)
specificity_post = round(TN / (TN + FP),2)
print(f"Pricision of Pruned Tree: {precision_post}")
print(f"Recall of Pruned Tree: {recall_post}")
print(f"F1 score of Pruned Tree: {f1_score_post}")
print(f"Specificity of Pruned Tree: {specificity_post}")

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_pruned_tree)
auc_tree = roc_auc_score(y_test, y_prob_pruned_tree)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Decision Tree(Pruned Tree) (AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Pruned Decision Tree')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("               Stratified KFold Decision Tree (Pruned)                  ")

n_splits = 3
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(pruned_tree, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')
print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_pruned = round(cv_results.mean(),2)
print(f"Mean AUC Score Pruned Decision Tree : {St_kfold_pruned}")


print("------------------------------------------------------------")
print("                      Logistic Regression                         ")
print("------------------------------------------------------------")

logistic_regression_model = LogisticRegression(random_state=5805)
logistic_regression_model.fit(X_train, y_train)

y_pred_logistic_regression = logistic_regression_model.predict(X_test)
y_prob_logistic_regression = logistic_regression_model.predict_proba(X_test)[::, -1]

print("               Performance Matrices Logistic Regression                  ")

accuracy_logistic_regression = round(accuracy_score(y_test, y_pred_logistic_regression),2)
confusion_matrix_logistic_regression = confusion_matrix(y_test, y_pred_logistic_regression)
recall_logistic_regression = round(recall_score(y_test, y_pred_logistic_regression),2)
roc_auc_logistic_regression = round(roc_auc_score(y_test, y_prob_logistic_regression),2)
sns.heatmap(confusion_matrix_logistic_regression, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
accuracy_logistic_regression = round(accuracy_score(y_test, y_pred_logistic_regression),2)
recall_logistic_regression = round(recall_score(y_test, y_pred_logistic_regression),2)


print("Metrics for Logistic Regression Model:")
print(f"Accuracy: {accuracy_logistic_regression}")
print("Confusion Matrix:")
print(confusion_matrix_logistic_regression)
print(f"Recall: {recall_logistic_regression}")
print(f"AUC: {roc_auc_logistic_regression}")


TP = confusion_matrix_logistic_regression[1][1]
TN = confusion_matrix_logistic_regression[0][0]
FP = confusion_matrix_logistic_regression[0][1]
FN = confusion_matrix_logistic_regression[1][0]

precision_lg = round(TP / (TP + FP),2)
recall_lg = round(TP / (TP + FN),2)
f1_score_lg = round(2 * (precision_lg * recall_lg) / (precision_lg + recall_lg),2)
specificity_lg = round(TN / (TN + FP),2)
print(f"Pricision of Logistic Regression: {precision_lg}")
print(f"Recall of Logistic Regression: {recall_lg}")
print(f"F1 score of Logistic Regression: {f1_score_lg}")
print(f"Specificity of Logistic Regression: {specificity_lg}")

print("               Stratified KFold Logistic Regression                  ")

n_splits = 3
stratified_kfold_pruned = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)

cv_results = cross_val_score(logistic_regression_model, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_lgr = round(cv_results.mean(),2)
print(f"Mean AUC Score Logistic Regression : {St_kfold_lgr}")

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_logistic_regression)
auc_tree = roc_auc_score(y_test, y_prob_logistic_regression)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Logistic Regression (AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic Regression')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print("------------------------------------------------------------")
print("                      Logistic Regression Grid Search                         ")
print("------------------------------------------------------------")

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'newton-cg']
}

grid_search = GridSearchCV(logistic_regression_model, param_grid, cv=5, scoring='roc_auc')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
y_prob_logistic_regression_gs = best_model.predict_proba(X_test)[::, -1]

accuracy_train = round(accuracy_score(y_train, y_pred_train),2)
accuracy_test = round(accuracy_score(y_test, y_pred_test),2)
roc_auc_logistic_regression_gs = round(roc_auc_score(y_test, y_prob_logistic_regression_gs),2)
confusion_matrix_logistic_gs = confusion_matrix(y_test, y_pred_pre)
sns.heatmap(confusion_matrix_logistic_gs, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Best Parameters:", best_params)
print("Accuracy on Training Set:", accuracy_train)
print("Accuracy on Testing Set:", accuracy_test)
print(f"AUC: {roc_auc_logistic_regression_gs}")
print(f"Confusion Matrix Logistic Regression : \n{confusion_matrix_logistic_gs}")


n_splits = 3
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)

cv_results = cross_val_score(best_model, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_lgrgs = round(cv_results.mean(),2)
print(f"Mean AUC Score Logistic regression GS : {St_kfold_lgrgs}")


fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_logistic_regression_gs)
auc_tree = roc_auc_score(y_test, y_prob_logistic_regression_gs)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Logistic Regression GS (AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic Regression (Grid Search )')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


TP = confusion_matrix_logistic_gs[1][1]
TN = confusion_matrix_logistic_gs[0][0]
FP = confusion_matrix_logistic_gs[0][1]
FN = confusion_matrix_logistic_gs[1][0]

precision_lg_gs = round(TP / (TP + FP),2)
recall_lg_gs = round(TP / (TP + FN),2)
f1_score_lg_gs = round(2 * (precision_lg_gs * recall_lg_gs) / (precision_lg_gs + recall_lg_gs),2)
specificity_lg_gs = round(TN / (TN + FP),2)
print(f"Pricision of Logistic Regression: {precision_lg_gs}")
print(f"Recall of Logistic Regression: {recall_lg_gs}")
print(f"F1 score of Logistic Regression: {f1_score_lg_gs}")
print(f"Specificity of Logistic Regression: {specificity_lg_gs}")


print("------------------------------------------------------------")
print("                      KNN                         ")
print("------------------------------------------------------------")

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train,y_train)
y_pred_train = knn_classifier.predict(X_train)
y_pred_test_knn = knn_classifier.predict(X_test)

y_prob_knn = knn_classifier.predict_proba(X_test)[::, -1]

print("               Performance Matrices KNN                  ")

accuracy_knn = round(accuracy_score(y_test, y_pred_test_knn),2)
confusion_matrix_knn = confusion_matrix(y_test, y_pred_test_knn)
recall_knn = round(recall_score(y_test, y_pred_test_knn),2)
roc_auc_knn = round(roc_auc_score(y_test, y_prob_knn),2)
sns.heatmap(confusion_matrix_knn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print("Metrics for KNN Model:")
print(f"Accuracy: {accuracy_knn}")
print("Confusion Matrix:")
print(confusion_matrix_knn)
print(f"Recall: {recall_knn}")
print(f"AUC: {roc_auc_knn}")

TP = confusion_matrix_knn[1][1]
TN = confusion_matrix_knn[0][0]
FP = confusion_matrix_knn[0][1]
FN = confusion_matrix_knn[1][0]

precision_knn = round(TP / (TP + FP),2)
recall_knn = round(TP / (TP + FN),2)
f1_score_knn = round(2 * (precision_knn * recall_knn) / (precision_knn + recall_knn),2)
specificity_knn = round(TN / (TN + FP),2)
print(f"Pricision of KNN: {precision_knn}")
print(f"Recall of KNN: {recall_knn}")
print(f"F1 score of KNN: {f1_score_knn}")
print(f"Specificity of KNN: {specificity_knn}")

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_knn)
auc_tree = roc_auc_score(y_test, y_prob_knn)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"KNN (AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for KNN')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

k_options = [i for i in range(1, 15)]
error_rate = []
for k in k_options:
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train,y_train)
    y_pred_test = knn_classifier.predict(X_test)
    error_rate.append(np.mean(y_test != y_pred_test))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 15, 1), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red',markersize=12)
plt.title('Error Rate vs. K Value')
plt.xticks(ticks=range(1, 15, 1))
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()



print("               Stratified KFold KNN                  ")

n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(knn_classifier, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_knn = round(cv_results.mean(),2)
print(f"Mean AUC Score: {St_kfold_knn}")

print("------------------------------------------------------------")
print("                      KNN Elbow Method                        ")
print("------------------------------------------------------------")

knn_classifier_ag = KNeighborsClassifier(n_neighbors=14)
knn_classifier_ag.fit(X_train,y_train)
y_pred_train = knn_classifier_ag.predict(X_train)
y_pred_test_ag = knn_classifier_ag.predict(X_test)

y_prob_knn_ag = knn_classifier_ag.predict_proba(X_test)[::, -1]
print("               Performance Matrices KNN Elbow Method                 ")

accuracy_knn_gs = round(accuracy_score(y_test, y_pred_test_ag),2)
confusion_matrix_knn_gs = confusion_matrix(y_test, y_pred_test_ag)
recall_knn = round(recall_score(y_test, y_pred_test_ag),2)
roc_auc_knn_gs = round(roc_auc_score(y_test, y_prob_knn_ag),2)
sns.heatmap(confusion_matrix_knn_gs, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Display the results
print("Metrics for KNN Model:")
print(f"Accuracy: {accuracy_knn_gs}")
print("Confusion Matrix:")
print(confusion_matrix_knn)
print(f"Recall: {recall_knn}")
print(f"AUC: {roc_auc_knn_gs}")

# print("------------------------------------------------------------")
# print("                      KNN Grid Search                        ")
# print("------------------------------------------------------------")
#
# param_grid = {
#     'n_neighbors': [4,8,10,14],
#     'weights': ['uniform', 'distance'],
#     'metric': ['euclidean']
# }
#
#
#
# grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='accuracy')
#
# grid_search.fit(X_train, y_train)
#
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)
#
# best_knn = grid_search.best_estimator_
#
# y_test_pred = best_knn.predict(X_test)
# # roc_auc_knn_gs = round(roc_auc_score(y_test, y_prob_knn_gs) ,2)
#
# TP = confusion_matrix_knn_gs[1][1]
# TN = confusion_matrix_knn_gs[0][0]
# FP = confusion_matrix_knn_gs[0][1]
# FN = confusion_matrix_knn_gs[1][0]
#
# precision_KNN_gs = round(TP / (TP + FP),2)
# recall_KNN_gs = round(TP / (TP + FN),2)
# f1_score_KNN_gs = round(2 * (precision_KNN_gs * recall_KNN_gs) / (precision_KNN_gs + recall_KNN_gs),2)
# specificity_KNN_gs = round(TN / (TN + FP),2)
# print(f"Pricision of KNN after GS: {precision_KNN_gs}")
# print(f"Recall of KNN after GS: {recall_KNN_gs}")
# print(f"F1 score of KNN after GS: {f1_score_KNN_gs}")
# print(f"Specificity of KNN after GS: {specificity_KNN_gs}")
#
# fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_knn_ag)
# auc_tree = roc_auc_score(y_test, y_prob_knn_ag)
#
# plt.figure(figsize=(8, 6))
# plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"KNN (AUC = {auc_tree:.2f})")
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC for KNN Grid Search')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.show()
#
#
# print("               Stratified KFold KNN Grid Search                 ")
#
#
# n_splits = 3
#
# stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
# cv_results = cross_val_score(knn_classifier_ag, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')
#
# print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
# St_kfold_knn_gs = round(cv_results.mean(),2)
# print(f"Mean AUC Score KNN GS: {St_kfold_knn_gs}")

print("------------------------------------------------------------")
print("                      Support Vector Machines                       ")
print("------------------------------------------------------------")

from sklearn.svm import SVC


print("                        Linear SVM                           ")

# Linear SVM
linear_svm = SVC(C=0.1, kernel='linear',probability=True)
linear_svm.fit(X_train, y_train)
y_pred_linear = linear_svm.predict(X_test)
y_prob_LinearSVM = linear_svm.predict_proba(X_test)[::, -1]

print("                        Poly SVM                           ")

# # Polynomial SVM
poly_svm = SVC(C=0.1, kernel='poly', probability=True)
poly_svm.fit(X_train, y_train)
y_pred_poly = poly_svm.predict(X_test)
y_prob_PolySVM = poly_svm.predict_proba(X_test)[::, -1]

print("                        RBF SVM                           ")

# RBF SVM
rbf_svm = SVC(C=0.1, kernel='rbf', probability=True)
rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)
y_prob_rbfSVM = rbf_svm.predict_proba(X_test)[::, -1]

print("                        Linear SVM Metrics                         ")

accuracy_linear = round(accuracy_score(y_test, y_pred_linear),2)
print(f'Accuracy: {accuracy_linear}')

# y_pred_prob = svm_classifier.predict_proba(X_test)[::, -1]
auc_score_linear = round(roc_auc_score(y_test, y_prob_LinearSVM),2)
print(f'AUC Score: {auc_score_linear}')

confusion_matrix_LSvm = confusion_matrix(y_test, y_pred_linear)
print(f"Confusion Matrix Linear SVM: \n{confusion_matrix_LSvm}")
sns.heatmap(confusion_matrix_LSvm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_LinearSVM)
auc_tree = roc_auc_score(y_test, y_prob_LinearSVM)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"SVM (AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Linear SVM')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


TP = confusion_matrix_LSvm[1][1]
TN = confusion_matrix_LSvm[0][0]
FP = confusion_matrix_LSvm[0][1]
FN = confusion_matrix_LSvm[1][0]

precision_LSvm = round(TP / (TP + FP),2)
recall_LSvm = round(TP / (TP + FN),2)
f1_score_LSvm = round(2 * (precision_LSvm * recall_LSvm) / (precision_LSvm + recall_LSvm),2)
specificity_LSvm = round(TN / (TN + FP),2)
print(f"Pricision of LinearS: {precision_LSvm}")
print(f"Recall of LinearS: {recall_LSvm}")
print(f"F1 score of LinearS: {f1_score_LSvm}")
print(f"Specificity of LinearS: {specificity_LSvm}")


n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(linear_svm, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_linearSvm = round(cv_results.mean(),2)
print(f"Mean AUC Score Linear SVM: {St_kfold_linearSvm}")


print("                        Polynomial SVM Metrics                          ")

accuracy_poly = round(accuracy_score(y_test, y_pred_poly),2)
print(f'Accuracy: {accuracy_poly}')

# y_pred_prob = svm_classifier.predict_proba(X_test)[::, -1]
auc_score_poly = round(roc_auc_score(y_test, y_prob_PolySVM),2)
print(f'AUC Score: {auc_score_poly}')

confusion_matrix_PSvm = confusion_matrix(y_test, y_pred_poly)
print(f"Confusion Matrix Poly SVM: \n{confusion_matrix_PSvm}")
sns.heatmap(confusion_matrix_PSvm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_PolySVM)
auc_tree = roc_auc_score(y_test, y_prob_PolySVM)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"SVM Poly(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Polynomial SVM')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


TP = confusion_matrix_PSvm[1][1]
TN = confusion_matrix_PSvm[0][0]
FP = confusion_matrix_PSvm[0][1]
FN = confusion_matrix_PSvm[1][0]

precision_polyS = round(TP / (TP + FP),2)
recall_polyS = round(TP / (TP + FN),2)
f1_score_polyS = round(2 * (precision_polyS * recall_polyS) / (precision_polyS + recall_polyS),2)
specificity_polyS = round(TN / (TN + FP),2)
print(f"Pricision of PolyS: {precision_polyS}")
print(f"Recall of PolyS: {recall_polyS}")
print(f"F1 score of PolyS: {f1_score_polyS}")
print(f"Specificity of PolyS: {specificity_polyS}")

n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(poly_svm, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_PolySvm = round(cv_results.mean(),2)
print(f"Mean AUC Score Polynomial SVM: {St_kfold_PolySvm}")

print("                        RBF SVM Metrics                           ")

accuracy_rbf = round(accuracy_score(y_test, y_pred_rbf),2)
print(f'Accuracy: {accuracy_rbf}')

auc_score_rbf = round(roc_auc_score(y_test, y_prob_rbfSVM),2)
print(f'AUC Score: {auc_score_rbf}')

confusion_matrix_RSvm = confusion_matrix(y_test, y_pred_rbf)
print(f"Confusion Matrix Poly SVM: \n{confusion_matrix_RSvm}")
sns.heatmap(confusion_matrix_RSvm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_rbfSVM)
auc_tree = roc_auc_score(y_test, y_prob_rbfSVM)

plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"SVM RBF(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for RBF SVM')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


TP = confusion_matrix_RSvm[1][1]
TN = confusion_matrix_RSvm[0][0]
FP = confusion_matrix_RSvm[0][1]
FN = confusion_matrix_RSvm[1][0]

precision_rbf = round(TP / (TP + FP),2)
recall_rbf = round(TP / (TP + FN),2)
f1_score_rbf = round(2 * (precision_rbf * recall_rbf) / (precision_rbf + recall_rbf),2)
specificity_rbf = round(TN / (TN + FP),2)
print(f"Pricision of RBF S: {precision_rbf}")
print(f"Recall of RBF S: {recall_rbf}")
print(f"F1 score of RBF S: {f1_score_rbf}")
print(f"Specificity of RBF S: {specificity_rbf}")

n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(rbf_svm, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_RBFSvm = round(cv_results.mean(),2)
print(f"Mean AUC Score RBF SVM: {St_kfold_RBFSvm}")
#
print("------------------------------------------------------------")
print("                      SVM Grid Search                        ")
print("------------------------------------------------------------")
svc = SVC(random_state=5805, probability=True)
svc_params = {'C': [0.4, 0.5],
              'kernel': ['poly', 'rbf']}
grid_svm = GridSearchCV(estimator=svc, param_grid=svc_params, scoring='f1_macro', n_jobs=-1)
grid_svm.fit(X_train, y_train)
print('After Grid Search best parameters are:')
best_params_svc = grid_svm.best_params_
best_estimator = grid_svm.best_estimator_
y_proba_svm_gs = best_estimator.predict_proba(X_test)[::, -1]

y_pred_svm_gs = best_estimator.predict(X_test)
accuracy_svm_gs = accuracy_score(y_test, y_pred_svm_gs)
auc_roc_svm_gs = roc_auc_score(y_test, best_estimator.predict_proba(X_test)[::, -1])

confusion_matrix_rf_gs = confusion_matrix(y_test, y_pred_svm_gs)
sns.heatmap(confusion_matrix_rf_gs, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

accuracy_rf_gs = round(accuracy_score(y_test, y_pred_svm_gs), 2)
print(f'Accuracy: {accuracy_rf_gs}')

auc_roc_svm_gss = round(roc_auc_score(y_test, y_proba_svm_gs), 2)
print(f'AUC Score: {auc_roc_svm_gss}')

confusion_matrix_svm_gs = confusion_matrix(y_test, y_pred_svm_gs)
print(f"Confusion Matrix Random Forest GS: \n{confusion_matrix_svm_gs}")

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_svm_gs)
auc_tree = roc_auc_score(y_test, y_proba_svm_gs)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"SVM GS (AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for SVM GS')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()



TP = confusion_matrix_svm_gs[1][1]
TN = confusion_matrix_svm_gs[0][0]
FP = confusion_matrix_svm_gs[0][1]
FN = confusion_matrix_svm_gs[1][0]

precision_svm_gs = round(TP / (TP + FP),2)
recall_svm_gs = round(TP / (TP + FN),2)
f1_score_svm_gs = round(2 * (precision_svm_gs * recall_svm_gs) / (precision_svm_gs + recall_svm_gs),2)
specificity_svm_gs = round(TN / (TN + FP),2)
print(f"Pricision of SVM GS: {precision_svm_gs}")
print(f"Recall of SVM GS: {recall_svm_gs}")
print(f"F1 score of SVM GS: {f1_score_svm_gs}")
print(f"Specificity of SVM GS: {specificity_svm_gs}")

print(best_params_svc)

print("------------------------------------------------------------")
print("                      Naive Bayes                       ")
print("------------------------------------------------------------")

from sklearn.naive_bayes import GaussianNB

naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

y_pred_NB = naive_bayes.predict(X_test)
y_prob_NB = naive_bayes.predict_proba(X_test)[::, -1]

print("               Performance Matrices Naive Bayes                 ")


accuracy_NB = round(accuracy_score(y_test, y_pred_NB),2)
print(f'Accuracy: {accuracy_NB}')

auc_score_NB = round(roc_auc_score(y_test, y_prob_NB),2)
print(f'AUC Score: {auc_score_NB}')

confusion_matrix_NB = confusion_matrix(y_test, y_pred_NB)
print(f"Confusion Matrix Naive Bayes: \n{confusion_matrix_NB}")
sns.heatmap(confusion_matrix_NB, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# classification_report_result = classification_report(y_test, y_pred_NB)


fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_NB)
auc_tree = roc_auc_score(y_test, y_prob_NB)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Naive Bayes(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Naive Bayes')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Confusion_matrix_of_NB = [[7996,814],[5750,3060]]

TP = confusion_matrix_NB[1][1]
TN = confusion_matrix_NB[0][0]
FP = confusion_matrix_NB[0][1]
FN = confusion_matrix_NB[1][0]

precision_NB = round(TP / (TP + FP),2)
recall_NB = round(TP / (TP + FN),2)
f1_score_NB = round(2 * (precision_NB * recall_NB) / (precision_NB + recall_NB),2)
specificity_NB = round(TN / (TN + FP),2)
print(f"Pricision of NB: {precision_NB}")
print(f"Recall of RBF NB: {recall_NB}")
print(f"F1 score of RBF NB: {f1_score_NB}")
print(f"Specificity of NB: {specificity_NB}")

print("               Stratified KFold Naive Bayes                 ")

n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(naive_bayes, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_NB = round(cv_results.mean(),2)
print(f"Mean AUC Score Naive Bayes : {St_kfold_NB}")

print("------------------------------------------------------------")
print("                      Naive Bayes Grid Search                      ")
print("------------------------------------------------------------")
param_grid = {
    'priors': [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]],
    'var_smoothing': np.logspace(0,-9, num=100)
}

naive_bayes_gs = GaussianNB()

scoring = {
    'Accuracy': make_scorer(accuracy_score),
    'AUC-ROC': make_scorer(roc_auc_score)
}

grid_search = GridSearchCV(naive_bayes, param_grid, cv=5, scoring=scoring, refit='AUC-ROC', verbose=2)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
y_proba_NB_gs = best_estimator.predict_proba(X_test)[::, -1]
print("Best Parameters:", best_params)

y_pred_NB_gs = best_estimator.predict(X_test)
accuracy_NB_gs = round(accuracy_score(y_test, y_pred_NB_gs),2)
auc_roc_NB_gs = round(roc_auc_score(y_test, best_estimator.predict_proba(X_test)[:, 1]),2)
confusion_matrix_NB_gs = confusion_matrix(y_test, y_pred_NB_gs)
sns.heatmap(confusion_matrix_NB_gs, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

print(f"Confusion Matrix Naive Bayes: \n{confusion_matrix_NB_gs}")
print(f"Accuracy on Test Set: {accuracy_NB_gs}")
print(f"AUC-ROC on Test Set: {auc_roc_NB_gs}")


TP = confusion_matrix_NB_gs[1][1]
TN = confusion_matrix_NB_gs[0][0]
FP = confusion_matrix_NB_gs[0][1]
FN = confusion_matrix_NB_gs[1][0]

precision_NB_gs = round(TP / (TP + FP),2)
recall_NB_gs = round(TP / (TP + FN),2)
f1_score_NB_gs = round(2 * (precision_NB * recall_NB) / (precision_NB + recall_NB),2)
specificity_NB_gs = round(TN / (TN + FP),2)
print(f"Pricision of NB: {precision_NB_gs}")
print(f"Recall of RBF NB: {recall_NB_gs}")
print(f"F1 score of RBF NB: {f1_score_NB_gs}")
print(f"Specificity of NB: {specificity_NB_gs}")

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_NB_gs)
auc_tree = roc_auc_score(y_test, y_proba_NB_gs)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Naive Bayes(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Naive Bayes Grid Search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


print("------------------------------------------------------------")
print("                      Random Forest                       ")
print("------------------------------------------------------------")

random_forest = RandomForestClassifier(n_estimators=50, random_state=5805)
random_forest.fit(X_train, y_train)

y_pred_rf = random_forest.predict(X_test)
y_prob_rf = random_forest.predict_proba(X_test)[::, -1]

print("               Performance Matrices Random Forest                ")

accuracy_rf = round(accuracy_score(y_test, y_pred_rf),2)
print(f'Accuracy: {accuracy_rf}')

auc_score_rf = round(roc_auc_score(y_test, y_prob_rf),2)
print(f'AUC Score: {auc_score_rf}')

confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
print(f"Confusion Matrix Poly SVM: \n{confusion_matrix_rf}")
sns.heatmap(confusion_matrix_rf, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_rf)
auc_tree = roc_auc_score(y_test, y_prob_rf)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Random Forest(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


TP = confusion_matrix_rf[1][1]
TN = confusion_matrix_rf[0][0]
FP = confusion_matrix_rf[0][1]
FN = confusion_matrix_rf[1][0]

precision_rf = round(TP / (TP + FP),2)
recall_rf = round(TP / (TP + FN),2)
f1_score_rf = round(2 * (precision_rf * recall_rf) / (precision_rf + recall_rf),2)
specificity_rf = round(TN / (TN + FP),2)
print(f"Pricision of RF: {precision_rf}")
print(f"Recall of RBF RF: {recall_rf}")
print(f"F1 score of RBF RF: {f1_score_rf}")
print(f"Specificity of RF: {specificity_rf}")

print("               Stratified KFold Random Forest                 ")

n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(random_forest, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_rf = round(cv_results.mean(),2)
print(f"Mean AUC Score Random Forest: {St_kfold_rf}")

print("------------------------------------------------------------")
print("                      Random Forest Grid Search                      ")
print("------------------------------------------------------------")

parameter_grid = {'n_estimators': [100, 125],
                  'criterion': ['gini', 'entropy']}
grid_rfc = GridSearchCV(estimator=random_forest, param_grid=parameter_grid, scoring='f1_macro',n_jobs=-1)
grid_rfc.fit(X_train, y_train)
best_params = grid_rfc.best_params_
best_estimator = grid_rfc.best_estimator_
y_proba_rf_gs = best_estimator.predict_proba(X_test)[::, -1]

y_pred_rf_gs = best_estimator.predict(X_test)
accuracy_NB_gs = accuracy_score(y_test, y_pred_rf_gs)
auc_roc_rf_gs = roc_auc_score(y_test, best_estimator.predict_proba(X_test)[::, -1])
confusion_matrix_rf_gs = confusion_matrix(y_test, y_pred_rf_gs)
sns.heatmap(confusion_matrix_rf_gs, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
print(f"Confusion Matrix Random Forest GS: \n{confusion_matrix_rf_gs}")

accuracy_rf_gs = round(accuracy_score(y_test, y_pred_rf),2)
print(f'Accuracy: {accuracy_rf_gs}')

auc_score_rf_gs = round(roc_auc_score(y_test, y_proba_rf_gs),2)
print(f'AUC Score: {auc_roc_rf_gs}')

# confusion_matrix_rf_gs = confusion_matrix(y_test, y_pred_rf)

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_proba_rf_gs)
auc_tree = roc_auc_score(y_test, y_proba_rf_gs)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Random Forest(GS)(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest Grid Search')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


TP = confusion_matrix_rf_gs[1][1]
TN = confusion_matrix_rf_gs[0][0]
FP = confusion_matrix_rf_gs[0][1]
FN = confusion_matrix_rf_gs[1][0]

precision_rf_gs = round(TP / (TP + FP),2)
recall_rf_gs = round(TP / (TP + FN),2)
f1_score_rf_gs = round(2 * (precision_rf_gs * recall_rf_gs) / (precision_rf_gs + recall_rf_gs),2)
specificity_rf_gs = round(TN / (TN + FP),2)
print(f"Pricision of RF GS: {precision_rf_gs}")
print(f"Recall of RBF RF GS: {recall_rf_gs}")
print(f"F1 score of RBF RF GS: {f1_score_rf_gs}")
print(f"Specificity of RF GS: {specificity_rf_gs}")

print("               Stratified KFold Random Forest                 ")


n_splits = 3
stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(random_forest, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_rf = round(cv_results.mean(),2)
print(f"Mean AUC Score Random Forest: {St_kfold_rf}")


print("               1. Bagging                 ")

bagging_classifier = BaggingClassifier(random_forest,random_state=5805)

bagging_classifier.fit(X_train, y_train)

y_pred_bagging = bagging_classifier.predict(X_test)
y_prob_bagging = bagging_classifier.predict_proba(X_test)[::, -1]

accuracy_bagging = round(accuracy_score(y_test, y_pred_bagging),2)
print(f'Accuracy: {accuracy_bagging}')

auc_score_bagging = round(roc_auc_score(y_test, y_prob_bagging),2)
print(f'AUC Score: {auc_score_bagging}')

confusion_matrix_bagging = confusion_matrix(y_test, y_pred_bagging)
print(f"Confusion Matrix Random Forest(Bagging): \n{confusion_matrix_bagging}")
sns.heatmap(confusion_matrix_bagging, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_bagging)
auc_tree = roc_auc_score(y_test, y_prob_bagging)


plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Bagging(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest(Bagging)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Confusion_matrix_of_bg = [[7881, 929],[1050, 7760]]

TP = confusion_matrix_bagging[1][1]
TN = confusion_matrix_bagging[0][0]
FP = confusion_matrix_bagging[0][1]
FN = confusion_matrix_bagging[1][0]

precision_Bg = round(TP / (TP + FP),2)
recall_Bg = round(TP / (TP + FN),2)
f1_score_Bg = round(2 * (precision_Bg * recall_Bg) / (precision_Bg + recall_Bg),2)
specificity_Bg = round(TN / (TN + FP),2)
print(f"Pricision of RF (Bagging): {precision_Bg}")
print(f"Recall of RF (Bagging): {recall_Bg}")
print(f"F1 score of RF (Bagging): {f1_score_Bg}")
print(f"Specificity of RF (Bagging): {specificity_Bg}")

n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(bagging_classifier, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_rfbg = round(cv_results.mean(),2)
print(f"Mean AUC Score Random Forest: {St_kfold_rfbg}")

print("               2. Stacking                 ")

estimators = [('BC', BaggingClassifier()),
              ('BO', AdaBoostClassifier(n_estimators=100)),
              ('GBO', GradientBoostingClassifier(n_estimators=100))]

stacking_classifier = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=5)
stacking_classifier.fit(X_train, y_train)

y_pred_stacking = stacking_classifier.predict(X_test)
y_prob_stacking = bagging_classifier.predict_proba(X_test)[::, -1]

accuracy_stacking = round(accuracy_score(y_test, y_pred_stacking),2)
print(f'Accuracy: {accuracy_stacking}')

auc_score_stacking = round(roc_auc_score(y_test, y_prob_stacking),2)
print(f'AUC Score: {auc_score_stacking}')

confusion_matrix_stacking = confusion_matrix(y_test, y_pred_stacking)
print(f"Confusion Matrix Poly SVM: \n{confusion_matrix_stacking}")
sns.heatmap(confusion_matrix_stacking, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_stacking)
auc_tree = roc_auc_score(y_test, y_prob_stacking)

plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Stacking(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest(Stacking)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Confusion_matrix_of_stacking = [[7880,  930],[ 966, 7844]]

TP = confusion_matrix_stacking[1][1]
TN = confusion_matrix_stacking[0][0]
FP = confusion_matrix_stacking[0][1]
FN = confusion_matrix_stacking[1][0]

precision_St = round(TP / (TP + FP),2)
recall_St = round(TP / (TP + FN),2)
f1_score_St = round(2 * (precision_St * recall_St) / (precision_St + recall_St),2)
specificity_St = round(TN / (TN + FP),2)
print(f"Pricision of RF (Stacking): {precision_St}")
print(f"Recall of RF (Stacking): {recall_St}")
print(f"F1 score of RF (Stacking): {f1_score_St}")
print(f"Specificity of RF (Stacking): {specificity_St}")

n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(stacking_classifier, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_rfst = round(cv_results.mean(),2)
print(f"Mean AUC Score Random Forest(Stacking): {St_kfold_rfst}")

print("               3. Boosting                 ")

adaboost_classifier = AdaBoostClassifier(random_forest,random_state=5805)
adaboost_classifier.fit(X_train, y_train)

y_pred_boosting = adaboost_classifier.predict(X_test)
y_prob_boosting = bagging_classifier.predict_proba(X_test)[::, -1]

accuracy_boosting = round(accuracy_score(y_test, y_pred_boosting),2)
print(f'Accuracy: {accuracy_boosting}')

auc_score_boosting = round(roc_auc_score(y_test, y_prob_boosting),2)
print(f'AUC Score: {auc_score_boosting}')

confusion_matrix_boosting = confusion_matrix(y_test, y_pred_boosting)
print(f"Confusion Matrix Boosting: \n{confusion_matrix_boosting}")
sns.heatmap(confusion_matrix_boosting, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_boosting)
auc_tree = roc_auc_score(y_test, y_prob_boosting)

plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"Boosting(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Random Forest(Boosting)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Confusion_matrix_of_boosting = [[7926, 884],[1136, 7674]]

TP = confusion_matrix_boosting[1][1]
TN = confusion_matrix_boosting[0][0]
FP = confusion_matrix_boosting[0][1]
FN = confusion_matrix_boosting[1][0]

precision_Bs = round(TP / (TP + FP),2)
recall_Bs = round(TP / (TP + FN),2)
f1_score_Bs = round(2 * (precision_Bs * recall_Bs) / (precision_Bs + recall_Bs),2)
specificity_Bs = round(TN / (TN + FP),2)
print(f"Pricision of RF (Boosting): {precision_Bs}")
print(f"Recall of RF (Boosting): {recall_Bs}")
print(f"F1 score of RF (Boosting): {f1_score_Bs}")
print(f"Specificity of RF (Boosting): {specificity_Bs}")

n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(adaboost_classifier, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_rfbs = round(cv_results.mean(),2)
print(f"Mean AUC Score Random Forest(Stacking): {St_kfold_rfbs}")

print("------------------------------------------------------------")
print("                      Neural Network (MLP)                       ")
print("------------------------------------------------------------")

from sklearn.neural_network import MLPClassifier

mlp_classifier = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=5805)

mlp_classifier.fit(X_train, y_train)

y_pred_nn = mlp_classifier.predict(X_test)
y_prob_nn = mlp_classifier.predict_proba(X_test)[::, -1]

accuracy = round(accuracy_score(y_test, y_pred_nn),2)
print(f"Accuracy Neural Network (MLP): {accuracy}")

confusion_matrix_NN = confusion_matrix(y_test, y_pred_nn)
print(f"Confusion Matrix Neural Network : \n{confusion_matrix_NN}")
sns.heatmap(confusion_matrix_NN, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
auc_score_NN = round(roc_auc_score(y_test, y_prob_nn),2)
print(f'AUC Score Neural Network (MLP): {auc_score_NN}')

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_nn)
auc_tree = roc_auc_score(y_test, y_prob_nn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"NN(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Neural Network(MLP)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.grid(True)
plt.show()

# Confusion_matrix_of_NN = [[7440, 1370],[1386, 7424]]

TP = confusion_matrix_NN[1][1]
TN = confusion_matrix_NN[0][0]
FP = confusion_matrix_NN[0][1]
FN = confusion_matrix_NN[1][0]

precision_NN = round(TP / (TP + FP),2)
recall_NN = round(TP / (TP + FN),2)
f1_score_NN = round(2 * (precision_NN * recall_NN) / (precision_NN + recall_NN),2)
specificity_NN = round(TN / (TN + FP),2)
print(f"Pricision of NN: {precision_NN}")
print(f"Recall of NN: {recall_NN}")
print(f"F1 score of NN: {f1_score_NN}")
print(f"Specificity of NN: {specificity_NN}")

n_splits = 3

stratified_kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=5805)
cv_results = cross_val_score(mlp_classifier, X_train, y_train, cv=stratified_kfold, scoring='roc_auc')

print(f"Stratified K-fold Cross-validation AUC Scores: {cv_results}")
St_kfold_mlp = round(cv_results.mean(),2)
print(f"Mean AUC Score Random MLP: {St_kfold_mlp}")

print("------------------------------------------------------------")
print("                      Neural Network (MLP) Grid Search                      ")
print("------------------------------------------------------------")
param_grid = {
    'hidden_layer_sizes': [(50,), (100,),(50,50), (100,100)],
    'activation': ['tanh', 'relu']
}

grid_search = GridSearchCV(mlp_classifier, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs = -1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print("Best Parameters Neural Network (MLP):", best_params)

y_pred_nn_gs = best_estimator.predict(X_test)
y_prob_nn_gs = mlp_classifier.predict_proba(X_test)[::, -1]

accuracy = accuracy_score(y_test, y_pred_nn_gs)
print(f"Accuracy Grid Search Neural Network (MLP): {accuracy}")
auc_score = roc_auc_score(y_test,y_prob_nn_gs)
print(f"Auc score Grid Search Neural Network (MLP): {accuracy}")

auc_score_NN_gs = round(roc_auc_score(y_test, y_prob_nn_gs),2)
print(f'AUC Score Neural Network (MLP): {auc_score_NN_gs}')

confusion_matrix_NN_gs = confusion_matrix(y_test, y_pred_nn_gs)
print(f"Confusion Matrix Neural Network : \n{confusion_matrix_NN_gs}")

sns.heatmap(confusion_matrix_NN_gs, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Confusion_matrix_of_NN_gs = [[7447, 1363],[1410, 7400]]

TP = confusion_matrix_NN_gs[1][1]
TN = confusion_matrix_NN_gs[0][0]
FP = confusion_matrix_NN_gs[0][1]
FN = confusion_matrix_NN_gs[1][0]

precision_NN_gs = round(TP / (TP + FP),2)
recall_NN_gs = round(TP / (TP + FN),2)
f1_score_NN_gs = round(2 * (precision_NN_gs * recall_NN_gs) / (precision_NN_gs + recall_NN_gs),2)
specificity_NN_gs = round(TN / (TN + FP),2)
print(f"Pricision of NN GS: {precision_NN_gs}")
print(f"Recall of NN GS: {recall_NN_gs}")
print(f"F1 score of NN GS: {f1_score_NN_gs}")
print(f"Specificity of NN GS: {specificity_NN_gs}")

fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_nn_gs)
auc_tree = roc_auc_score(y_test, y_prob_nn_gs)

plt.figure(figsize=(8, 6))
plt.plot(fpr_tree, tpr_tree,color='darkorange', label=f"NN (GS)(AUC = {auc_tree:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Neural Network(MLP) Grid Search')
plt.legend(loc="lower right")
plt.tight_layout()
plt.grid(True)
plt.show()

#
print("------------------------------------------------------------")
print("                      Classifiers' Performance Matrices Table                        ")
print("------------------------------------------------------------")

from prettytable import PrettyTable

newTable = PrettyTable(["Classifier", "Confusion Matrix", "Precision", "Recall","Specificity","F1 score","ROC-AUC"])

newTable.add_row(["Decision Tree", confusion_matrix_basic, precision, recall, specificity, f1_score, basic_tree_roc_auc])

newTable.add_row(["Pre-pruned Tree", confusion_matrix_pre, precision_pre, recall_pre, specificity_pre, f1_score_pre, roc_auc_pre])

newTable.add_row(["Pruned Tree", confusion_matrix_pruned, precision_post, recall_post, specificity_post, f1_score_post, roc_auc_pruned])

newTable.add_row(["Logistic Regression", confusion_matrix_logistic_regression, precision_lg, recall_lg, specificity_lg, f1_score_lg, roc_auc_logistic_regression])

newTable.add_row(["Logistic Grid Search", confusion_matrix_logistic_gs, precision_lg_gs, recall_lg_gs, specificity_lg_gs, f1_score_lg_gs, roc_auc_logistic_regression_gs])

newTable.add_row(["KNN ", confusion_matrix_knn, precision_knn, recall_knn, specificity_knn, f1_score_knn, roc_auc_knn])

# newTable.add_row(["KNN Grid Search", confusion_matrix_knn_gs, precision_KNN_gs, recall_KNN_gs, specificity_KNN_gs, f1_score_KNN_gs, roc_auc_knn_gs])

newTable.add_row(["Linear SVM", confusion_matrix_LSvm, precision_LSvm, recall_LSvm, specificity_LSvm, f1_score_LSvm, auc_score_linear])

newTable.add_row(["Polynomial SVM", confusion_matrix_PSvm, precision_polyS, recall_polyS, specificity_polyS, f1_score_polyS, auc_score_poly])

newTable.add_row(["RBF SVM", confusion_matrix_RSvm, precision_rbf, recall_rbf, specificity_rbf, f1_score_rbf, auc_score_rbf])

newTable.add_row(["SVM Grid Search", confusion_matrix_rf_gs, precision_svm_gs, recall_svm_gs, specificity_svm_gs, f1_score_svm_gs, auc_roc_svm_gss])

newTable.add_row(["Naive Bayes", confusion_matrix_NB, precision_NB, recall_NB,specificity_NB,f1_score_NB,auc_score_NB])

newTable.add_row(["Naive Bayes Grid search", confusion_matrix_NB_gs , precision_NB_gs, recall_NB_gs, specificity_NB_gs, f1_score_NB_gs, auc_roc_NB_gs])

newTable.add_row(["Random Forest", confusion_matrix_rf, precision_rf, recall_rf, specificity_rf, f1_score_rf ,auc_score_rf])

newTable.add_row(["Random Forest Grid Search", confusion_matrix_rf_gs, precision_rf_gs, recall_rf_gs, specificity_rf_gs, f1_score_rf_gs, auc_score_rf_gs])

newTable.add_row(["Bagging (RF)", confusion_matrix_bagging, precision_Bg, recall_Bg, specificity_Bg, f1_score_Bg, auc_score_bagging])

newTable.add_row(["Stacking (RF)", confusion_matrix_stacking, precision_St, recall_St, specificity_St, f1_score_St, auc_score_stacking])

newTable.add_row(["Boosting (RF)", confusion_matrix_boosting, precision_Bs, recall_Bs, specificity_Bs, f1_score_Bs, auc_score_boosting])

newTable.add_row(["Neural Network (MLP)", confusion_matrix_NN, precision_NN, recall_NN, specificity_NN, f1_score_NN, auc_score_NN])

newTable.add_row(["Neural Network Grid Search(MLP)", confusion_matrix_NN_gs, precision_NN_gs, recall_NN_gs, specificity_NN_gs, f1_score_NN_gs, auc_score_NN_gs])

print(newTable)


