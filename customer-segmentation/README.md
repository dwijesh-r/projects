# Customer-Segmentation-for-Targeted-Marketing-Insights

This repository has the code for the project which clusters customers based on RFM.

Aim: To cluster the user data to understand various categories of users and type of marketing schemes we can utilize for each cluster.

Software Used: Python, Tableau

Approach:

Step 1: **Exploratory Data Analysis**

As part of the EDA, analyzed te summary statistics of the data to find out well distributed data columns. Found out the presence of outliers and null values; either filtered them or imputed them based on the requirement.

Step 2: **Variable Selection**

In order to continue with the analysis we need to decide a few columns for our study. We can not use all the columns as part of our data analysis, it wouldn't be a good approach to use all the data in the analysis since that might make the analysis biased or even overfitted models.

In our case we have considered a few variables for clustering and a few for demographic analysis.

Step 3: **K-Means Clustering**

We used K-Means Clustering to cluster the data. We used the K-Means Elbow curve to find the nuber of clusters. Our case had 3 clusters to find the optimum clustering. The clustering was done on the basis of Recency, Frequency and Monetary.

Step 4: **Cluster Profiling and Outcomes**

Cluster 0: Low-Spending, Infrequent Customers
Cluster 1: High-Spending, Frequent Customers
Cluster 2: Moderate Spending, Frequent Customers

Please refer to the additional files in the repository for the code and an indepth analysis of the demographics through visualizations.
