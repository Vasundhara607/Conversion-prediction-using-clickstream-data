# Conversion-prediction-using-clickstream-data

In modern e-commerce, predicting whether a user will make a purchase (convert) is critical for personalized marketing and revenue optimization. This project implements a complete AI-driven workflow to predict session conversions using a dataset of 179,656 event-level observations.

---


The core innovation of this project is the integration of Deep Learning (LSTM) with Explainable AI (XAI) to ensure that model predictions are not only accurate but also transparent and interpretable for business stakeholders.


---

## Dataset splitting

1)Originally, the dataset contained over 2.7 million clickstream events capturing user interactions such as views, add-to-cart actions, and transactions. After extensive data cleaning, session-safe preprocessing, and leakage prevention, the dataset was significantly reduced to retain only valid behavioral signals for modeling. 

2)During exploratory analysis, we examined the class distribution and confirmed that transaction events were relatively rare compared to non-conversion interactions, highlighting a clear class imbalance problem common in e-commerce conversion datasets. 

3)To properly train and evaluate the models while avoiding data leakage, the processed sessions were divided into training, validation, and test sets using a 5: 2: 1 ratio, ensuring that model development, tuning, and final evaluation were performed on separate and unbiased subsets of the data.

---

**Tech Stack**

* **Languages**: R (Primary), Python (via reticulate integration) 

* **Deep Learning**: TensorFlow, Keras (LSTM architecture) 

* **Machine Learning**: glmnet (LASSO Regression) 

* **Explainable AI**: iml package (Shapley Values, Feature Importance) 





