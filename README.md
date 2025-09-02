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
 | ID  | Feature Name                          | Feature Description                                                                       |
|-----|---------------------------------------|--------------------------------------------------------------------------------------------|
| 1   | Timestamp                             | Date/Time                                                                                  |
| 2   | age                                   | Age                                                                                        |
| 3   | Gender                                | Gender                                                                                     |
| 4   | Purchase_Frequency                    | How frequently do you make purchases on Amazon?                                           |
| 5   | Purchase_Categories                   | What product categories do you typically purchase on Amazon?                              |
| 6   | Personalized_Recommendation_Frequency | Have you ever made a purchase based on personalized product recommendations from Amazon?  |
| 7   | Browsing_Frequency                    | How often do you browse Amazon's website or app?                                          |
| 8   | Product_Search_Method                 | How do you search for products on Amazon?                                                 |
| 9   | Search_Result_Exploration             | Do you explore multiple pages of search results or just the first one?                    |
| 10  | Customer_Reviews_Importance           | How important are customer reviews in your decision-making process?                       |
| 11  | Add_to_Cart_Browsing                  | Do you add products to your cart while browsing on Amazon?                                |
| 12  | Cart_Completion_Frequency             | How often do you complete the purchase after adding products to your cart?                |
| 13  | Cart_Abandonment_Factors              | What factors influence your decision to abandon a purchase in your cart?                  |
| 14  | Saveforlater_Frequency                | Do you use Amazon's "Save for Later" feature, and if so, how often?                       |
| 15  | Review_Left                           | Have you ever left a product review on Amazon?                                             |
| 16  | Review_Reliability                    | How much do you rely on product reviews when making a purchase?                          |
| 17  | Review_Helpfulness                    | Do you find helpful information from other customers' reviews?                            |
| 18  | Personalized_Recommendation_Frequency | How often do you receive personalized product recommendations from Amazon?                |
| 19  | Recommendation_Helpfulness            | Do you find the recommendations helpful?                                                  |

  
### Detailed Insights
###Understanding the Task
###Engineering Features
###Baseline Model
###Simple Model/Score the Model
###Improving the Model
###Next Steps and Recommendations
