# Purchase Prediction of Books - Customer Segmentation and Targeted Marketing

This repository contains code and information regarding the implementation of machine learning algorithms to predict customer purchases for a bookstore (referred to as CBC in this documentation). The goal is to optimize targeted marketing strategies to reach the most profitable customers and prospects.

## Objective
The objective of this project is to leverage customer data to predict which customers are likely to contribute to sales, enabling CBC to design and implement targeted marketing campaigns effectively.

## Implementation Details
### Customer Acquisition
- Direct Mailing: Promotional materials and offers are sent directly to existing club members via postal mail.
- Telemarketing: Sales calls are made to existing club members to promote products or solicit feedback.

### Data Collection
- All customer responses are recorded and maintained in a centralized database.
- Critical missing information is requested directly from the customer to ensure comprehensive data collection.

### Machine Learning Models
- Logistic Regression: Implemented to predict customer purchases.
  - Achieved an accuracy score of 92.583% on the provided dataset.
  - Evaluated using ROC curve and confusion matrix, showing good performance.
  - Features selected based on analysis of histogram plots, box plots, heatmap for individual variability, joint variability, and correlation values.
