Amazon Delivery Time Prediction
Project Overview

The Amazon Delivery Time Prediction project aims to estimate delivery times for e-commerce orders based on various factors such as distance, traffic, weather, product category, and agent performance. This project leverages machine learning regression models to provide accurate predictions and deploys a user-friendly interface for real-time delivery time estimation.

Features

Predict delivery times using multiple regression models.

Analyze key factors affecting delivery performance, including traffic, weather, and distance.

Evaluate agent efficiency and operational trends.

Interactive web application for inputting order details and getting predicted delivery times.

Model tracking and comparison using MLflow.

Skills and Tools

Python scripting

Data Cleaning and Preprocessing

Exploratory Data Analysis (EDA)

Feature Engineering

Machine Learning Regression Models: Linear Regression, Random Forest, Gradient Boosting

MLflow for model tracking

Streamlit for building the interactive web application

Libraries: pandas, numpy, scikit-learn, xgboost, seaborn, matplotlib, geopy

Dataset

The project uses the amazon_delivery.csv dataset, which contains information about orders, delivery agents, and delivery conditions.

Key columns:

Order_ID: Unique identifier for each order

Agent_Age, Agent_Rating

Store_Latitude, Store_Longitude, Drop_Latitude, Drop_Longitude

Order_Date, Order_Time, Pickup_Time

Weather, Traffic, Vehicle, Area

Category: Product category

Delivery_Time: Target variable representing actual delivery time (hours)

Project Steps
1. Data Preparation

Load the dataset and inspect the data.

Handle missing values and duplicates.

Standardize categorical variables.

2. Exploratory Data Analysis (EDA)

Analyze distribution of delivery times.

Study the effect of traffic, weather, and distance.

Visualize agent performance and delivery trends.

3. Feature Engineering

Calculate distance between store and delivery locations using geospatial coordinates.

Extract time-based features such as hour of day and day of week.

Encode categorical variables for model compatibility.

4. Model Development

Split data into training and testing sets.

Train regression models: Linear Regression, Random Forest, Gradient Boosting.

Evaluate models using MAE, RMSE, and R-squared.

Track models and their performance metrics using MLflow.

5. Application Development

Build a Streamlit interface for real-time delivery time prediction.

Input features such as distance, traffic, weather, and agent rating.

Display predicted delivery time to users.

6. Deployment

Deploy the Streamlit application for web access.

Ensure scalability and accessibility for users.
