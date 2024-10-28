# Airbnb-Price-Prediction-and-Host-Classification

This project utilizes machine learning techniques to analyze Airbnb listings with the goal of predicting property prices and classifying hosts as "superhosts" or "regular hosts." 
The dataset used is the publicly available Airbnb Open Data from Kaggle, which includes diverse features such as property attributes, neighborhood information, and host details. 
The project demonstrates regression and classification methods, clustering, and association rule mining to offer insights into price determinants and host categorization.

Dataset

The Airbnb dataset includes:

  1. Property Attributes: Type (e.g., apartment, house), number of bedrooms, bathrooms, amenities.
  2. Host Attributes: Host ID, identity verification, and superhost status.
  3. Location Data: Neighborhood and geographical coordinates.
  4. Availability and Pricing: Nightly price, cleaning fees, minimum nights, and number of reviews.

Objectives

  1. Price Prediction (Regression): Predict listing prices based on property and host attributes.
  2. Host Classification: Categorize hosts as either "superhosts" or "regular hosts."
  3. Clustering: Segment properties based on booking and review attributes.
  4. Association Rule Mining: Explore patterns between booking attributes and price categories.

Methodology
  1. Data Preprocessing: Data cleaning, handling missing values, and encoding categorical variables.
  2. Feature Engineering: Created new features such as Host_type (superhost vs. normal) and price_category (standard vs. expensive).
  3. Regression Analysis: Employed Linear Regression for price prediction.
  4. Classification Analysis: Used classifiers including Decision Trees, Random Forests, SVM, K-Nearest Neighbors, and Neural Networks to classify host types.
  5. Clustering: Applied K-Means clustering for property segmentation.
  6. Association Rule Mining: Conducted association rule mining using Apriori algorithm to analyze instant booking and pricing patterns.

Results

  1. Price Prediction:

      a. R2 Score: Achieved an R2 of around 0.7, indicating a good fit for the regression model.

  2. Classification (Host Type):

      b. The Random Forest classifier achieved an accuracy of 88% with a ROC-AUC of 0.95, making it the best-performing model.
     
  3. Clustering:
     
      c. Using Silhouette analysis, the optimal number of clusters was identified, offering insights into property groupings.
     
  4. Association Rule Mining:
     
      d. Generated rules to determine patterns, such as "If instant_bookable is False, the accommodation is likely to be expensive."


