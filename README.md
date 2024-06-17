**Project Overview**
This project is a comprehensive analysis of a telecom dataset, focusing on user overview, engagement, experience, and satisfaction. 
The goal is to provide insights into customer behavior, identify areas of improvement, and develop a predictive model for customer satisfaction.

**Task 1: User Overview Analysis**

Conducted exploratory data analysis to understand the dataset and identify missing values and outliers.
Identified top 10 handsets used by customers, top 3 handset manufacturers, and top 5 handsets per manufacturer.
Made recommendations to marketing teams based on the findings.

**Task 2: User Engagement Analysis**

Tracked user engagement using sessions frequency, session duration, and session total traffic.
Aggregated engagement metrics per customer and reported top 10 customers per metric.
Normalized metrics and performed k-means clustering to classify customers into three engagement groups.
Computed minimum, maximum, average, and total non-normalized metrics for each cluster.

**Task 3: Experience Analytics**

Focused on network parameters (TCP retransmission, Round Trip Time, Throughput) and customer device characteristics (handset type).
Aggregated average TCP retransmission, RTT, handset type, and throughput per customer.
Computed and listed top, bottom, and most frequent values for each network parameter.
Performed k-means clustering to segment users into groups of experiences.

**Task 4: Satisfaction Analysis**

Assigned engagement and experience scores to each user based on Euclidean distance.
Calculated satisfaction scores as the average of engagement and experience scores.
Reported top 10 satisfied customers and built a regression model to predict satisfaction scores.
Performed k-means clustering on engagement and experience scores and aggregated average satisfaction and experience scores per cluster.

**Model Deployment Tracking**

Deployed the model using Streamlit, including code version, start and end time, source, parameters, metrics, and artifacts.


**Prepared by**
Vicky Saini
