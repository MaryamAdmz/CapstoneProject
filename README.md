# Capstone Project
This project aims to classify Amazon customers based on their add-to-cart behavior, specifically whether they add products while browsing (Yes), do not add any (No), or are uncertain (Maybe), using their browsing activity and survey responses. To achieve this, we apply machine learning models, including
 - Logistic Regression
 - K-Nearest Neighbor(KNN)
 - Decision Trees
 - Support Vector Machines(SVM)
 - XGBoost
 - GaussianNB
 - Random Forest.

The dataset is designed to provide insights into customer preferences, shopping habits, and decision-making processes. By analyzing it, researchers and analysts can gain a deeper understanding of consumer behavior, identify emerging trends, optimize marketing strategies, and enhance the overall customer experience on Amazon.

## Understanding the Data
This dataset originates from [Kaggle](https://www.kaggle.com/datasets/swathiunnikrishnan/amazon-consumer-behaviour-dataset) and was collected through a survey on the Amazon ecosystem.
 - **Size:** 603 observations with 23 features
 - **Label:**
    - "Yes": customer added item(s)
    - "No": customer browsing only
    - "Maybe": customer is uncertain
 - **Features/Descriptions**
   
| Feature Name                          | Description                                                  |
|---------------------------------------|--------------------------------------------------------------|
| Timestamp                             | Date and time of the response                                |
| age                                   | Respondent's age                                             |
| Gender                                | Respondent's gender                                          |
| Purchase_Frequency                    | Frequency of purchases on Amazon                             |
| Purchase_Categories                   | Common product categories purchased                          |
| Personalized_Recommendation_Frequency | Bought from personalized recommendations?                    |
| Browsing_Frequency                    | How often user browses Amazon                                |
| Product_Search_Method                 | How the user searches for products                           |
| Search_Result_Exploration             | Tendency to explore beyond first page                        |
| Customer_Reviews_Importance           | Importance of customer reviews                               |
| Add_to_Cart_Browsing                  | Adds items to cart while browsing?                           |
| Cart_Completion_Frequency             | Frequency of completing purchases                            |
| Cart_Abandonment_Factors              | Reasons for abandoning cart                                  |
| Saveforlater_Frequency                | Uses “Save for Later” feature?                               |
| Review_Left                           | Left a review on Amazon?                                     |
| Review_Reliability                    | Trust in customer reviews                                    |
| Review_Helpfulness                    | Finds other reviews helpful?                                 |
| Recommendation_Helpfulness            | Finds recommendations helpful?                                       |
|Rating_Accuracy	                       |How would you rate the accuracy of the recommendations you receive    |
|Shopping_Satisfaction                 	|How satisfied are you with your overall shopping experience on Amazon?|
|Service_Appreciation	                  |What aspects of Amazon's services do you appreciate the most?|
|Improvement_Areas                      |	Are there any areas where you think Amazon can improve?|


  
### Detailed Insights
The dataset was examined for missing values, duplicates, and outliers: 
  - Only 2 null values were found in the Product_search_Method column; they were replaced with the mode of that column.
  - The dataset contains a duplicated column, Personalized_Recommendation_Frequency, once as an object type and once as int64.
  - No outliers were detected in the dataset.
Based on my analysis of the dataset, several interesting patterns emerged about how customers interact with Amazon. These insights focus on age, gender, product preferences, and browsing habits.
- Age Group Activity:
 - Young users aged 0–20 are the most active on the platform. They interact with the website more than other age groups.
- Gender Breakdown:
  - Most customers are female (58.5%), followed by male (23.6%), prefer not to say (14.8%), and others (3.16%).
- Popular Product Categories:
 The top three categories are: Clothing and Fashion, Beauty and Personal Care, and Others.
In comparison, categories like Groceries and Home and Kitchen are less popular.
- What Customers Like:
People appreciate: Product recommendations, Good prices, A wide selection of items
- What Needs Improvement:
Customers want better: Customer service, Product quality and accuracy, and Less packaging waste.
- Browsing Patterns: Female users tend to browse more during the evening and night.

- Looking at Multiple Factors Together:
 - When we look at age, gender, time, and add-to-cart behavior: Female adults and young adults are the main group who browse without buying, especially at night. The same group also tends to make purchases at night more than others.






###Understanding the Task
###Engineering Features
###Baseline Model
###Simple Model/Score the Model
###Improving the Model
###Next Steps and Recommendations
