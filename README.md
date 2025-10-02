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
    
Based on my analysis of the dataset, several interesting patterns emerged about how customers interact with Amazon. These insights individually focus on age, gender, product preferences, and browsing habits:
- Age Group Activity: Young users aged 0–20 are the most active on the platform. They interact with the website more than other age groups.
- Gender Breakdown: Most customers are female (58.5%), followed by male (23.6%), prefer not to say (14.8%), and others (3.16%).
- Popular Product Categories:
 The top three categories are: Clothing and Fashion, Beauty and Personal Care, and Others.
In comparison, categories like Groceries and Home and Kitchen are less popular.
- What Customers Like:
People appreciate: Product recommendations, Good prices, A wide selection of items
- What Needs Improvement:
Customers want better: Customer service, Product quality and accuracy, and Less packaging waste.
- Browsing Patterns: Female users tend to browse more during the evening and night.

<p align="center">
  <img src="Figures/plot4.png" alt=" " width="300"/>
  <img src="Figures/plot5.png" alt=" " width="300"/>
  <img src="Figures/plot6.png" alt=" " width="300"/>
</p>

<p align="center">
  <img src="Figures/plot1.png" alt=" " width="300"/>
  <img src="Figures/plot2.png" alt=" " width="300"/>
  <img src="Figures/plot3.png" alt=" " width="300"/>
</p>





### Combined Insight: Age, Gender, Time, and Purchase Behavior
When we look at age, gender, time, and add-to-cart behavior, Female adults and young adults are the main group who browse without buying, especially at night. The same group also tends to make purchases at night more than others.
![plot8](Figures/plot8.png)
## Understanding the Task
The goal of this machine learning project is to identify the key factors that influence whether a customer adds an item to their cart on Amazon. By understanding the patterns behind this behavior, we aim to improve product engagement and conversion strategies.

We seek to answer questions such as:
- Do age, gender, and certain times of day affect add-to-cart behavior?
- Do frequent shoppers add to cart more often?
- Which product categories lead to more cart activity?
- Are personalized recommendations linked to add-to-cart actions?
- Does the way customers interact with reviews impact their purchase intent?

## Engineering Features
The following steps were taken to prepare the data for modeling:

- Feature Selection:
   The top 10 features selected using SelectKBest with ANOVA F-test include a mix of encoded categorical and numerical    variables such as Gender, Time_Status, Age_Category, Browsing_Frequency, Personalized_Recommendation_Frequency,    Recommendation_Helpfulness, and Shopping_Satisfaction, among others.
  
- Preprocessing with ColumnTransformer:
  - Numerical features were passed through without modification.
  - Categorical features were transformed using OneHotEncoder to handle non-numeric values appropriately.
- Target Encoding:
The target variable (Add_to_Cart_Browsing) was encoded into numeric labels using pandas.Categorical.
- Data Splitting:
The dataset was split into training and testing sets using train_test_split, with 80% of the data used for training and 20% reserved for testing.


## Baseline Model
For the baseline model, we chose Logistic Regression, a simple yet effective method that can handle multi-class classification. 
After training and testing the model on the dataset, the following results were observed:
| Model Name          | Accuracy   | Precision  | Recall 	   | F1_Score   | 
|---------------------|:-----------|:-----------|:-----------|:-----------|
| Logestic Regression | 0.6362     | 0.7240     | 0.6924    |  0.6995    | 

This shows that the model performs consistently across accuracy, precision, recall, and F1-score, without strongly favoring either class. The similar values across metrics suggest the model is fairly balanced. However, there’s still room to improve, and further tuning, better feature engineering, or different models, such as decision trees, random forests, or XGBoost, could help increase its performance.

## Improving the Model
To improve model performance, the following strategies were applied
  - Class balancing with SMOTE: This technique was used to address imbalance across the three target classes, enabling models to learn from underrepresented groups more effectively.

- Model experimentation: Multiple classification models such as KNN, SVM, Decision Tree, Random Forest, GaussianNB, XGBoost were tested to compare performance and identify the most suitable algorithm for the task.

- Hyperparameter tuning: RandomizedSearchCV was used to optimize key model parameters such as the number of neighbors, depth, and learning rate to improve generalization and reduce overfitting.

These improvements collectively led to more balanced and accurate predictions across all three target classes.

Results before hyperparameter tuning
|Model Name           |Train Accuracy   | Test Accuracy  | Train Time(s) |
|---------------------|:----------------|:---------------|:--------------|
| knn                 | 0.6383          | 0.5785         | 0.0510        |
| SVM                 | 0.6798          | 0.6612         | 0.0448        |
| Decision Tree       | 0.6965          | 0.6364         | 0.0382        |
| GaussianNB          | 0.6466          | 0.6942         | 0.0354        |
| Random Forest       | 0.6965          | 0.6364         | 0.3709        |
| XGBoost             | 0.6965          | 0.6612         | 0.2280        |

Results after hyperparameter tuning
| Model Name          | Train Accuracy   |Test Accuracy  | Train Time(s) |
|---------------------|:-----------------|:--------------|:--------------|
| knn                 | 0.6985           | 0.6364        | 1.55          |
| SVM                 | 0.6611           | 0.7190        | 2.78          |
| Decision Tree       | 0.6736           | 0.6446        | 1.69          |
| GaussianNB          | 0.6071           | 0.6364        | 2.54          |
| Random Forest       | 0.7339           | 0.7190        | 15.95         |
| XGBoost             | 0.7900           | 0.6612        | 7.10          |

Model Performance Analysis:
- Before hyperparameter tuning, the strongest performer was GaussianNB, which achieved the highest test accuracy (0.6942) despite a relatively modest training accuracy (0.6466). Other models such as SVM (0.6612) and XGBoost (0.6612) performed reasonably, while Decision Tree and Random Forest both settled at 0.6364 test accuracy. KNN remained the weakest with 0.5785 test accuracy.

- After hyperparameter tuning, performance improved significantly. Random Forest and SVM both reached the top test accuracy of 0.7190, but their training performance tells an important story:

   - Random Forest achieved 0.7339 train accuracy and 0.7190 test accuracy, showing only a small gap between training and testing performance. This indicates a good fit with relatively low overfitting.

    - SVM, by contrast, had 0.6611 train accuracy and 0.7190 test accuracy. While it generalized well, the notably lower training score suggests that it underfit the data compared to Random Forest.

 - XGBoost reached the highest training accuracy (0.7900), but its test accuracy plateaued at 0.6612, a clear sign of overfitting given the limited dataset size.

   - Other models showed modest improvements but did not match the performance of Random Forest and SVM. Notably, GaussianNB declined in both train and test accuracy after tuning.

Conclusion:
Considering both training and testing accuracy, Random Forest is the most reliable model for this dataset. It offers strong test performance while maintaining balanced training accuracy, avoiding the underfitting risk seen in SVM and the overfitting problem of XGBoost.




## Next Steps and Recommendations


Despite its usefulness, the dataset has several important drawbacks:
- The dataset covers only a 13-day period, which limits its ability to capture seasonal patterns such as holidays, sales events, or weekday vs weekend variations. This short window introduces temporal bias and reduces the representativeness of long-term consumer behavior.
- The sample size is relatively small (602 records), which reduces statistical power and makes it harder to build robust, generalizable models. Small datasets are also more vulnerable to noise and overfitting.
- Being survey-based, the data may contain response bias (such as socially desirable answers) and imprecise categories such as "Sometimes" or "Few times a month", which weaken the accuracy of the analysis.


