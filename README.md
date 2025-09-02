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
 - **Features/Description**
   
| Feature Name                          | Description                                                  |
|---------------------------------------|--------------------------------------------------------------|
| Timestamp                             | Date and time of the response                                |
| age                                   | Respondent's age                                             |
| Gender                                | Respondent's gender                                          |
| Purchase_Frequency                    | Frequency of purchases on Amazon                             |
| Purchase_Categories                   | Common product categories purchased                          |
| Personalized_Recommendation_Frequency | Bought from personalized recommendations?                    |
| Browsing_Frequency                    | How often user browses Amazon                                |
| Product_Search_Method                 | How user searches for products                               |
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
|Rating_Accuracy	                       |How would you rate the relevance/accuracy of the recom you receive    |
|Shopping_Satisfaction                 	|How satisfied are you with your overall shopping experience on Amazon?|
|Service_Appreciation	                  |What aspects of Amazon's services do you appreciate the most?|
|Improvement_Areas                      |	Are there any areas where you think Amazon can improve?|


  
### Detailed Insights
###Understanding the Task
###Engineering Features
###Baseline Model
###Simple Model/Score the Model
###Improving the Model
###Next Steps and Recommendations
