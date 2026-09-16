<h1> Olist Late Delivery Prediction </h1>

## 1. Project Overview

This project develops a machine learning solution designed to predict late delivery risk for Olist (Brazil’s largest e-commerce marketplace aggregator), explain the operational drivers of shipping delays, and optimize proactive seller intervention workflows. In a marketplace model connecting 3,000+ independent merchants across 27 Brazilian states, late deliveries collapse customer review scores from  4.21 down to 2.26 stars, directly threatening seller conversion rates and platform revenue.

Standard evaluation metrics (e.g., Accuracy, ROC-AUC) fail under severe class imbalance (6.5% late rate), ignoring the business reality that a False Negative (missed late order) incurs massive reactive support and compensation costs, whereas a False Positive (automated alert) costs are minimum. To address this asymmetry, this project combines an optimized classification pipeline with Post Training Threshold Tuning and a Cost Benefit Analysis (CBA) based on published Brazilian business messaging rates and cross-industry customer support benchmarks.

### Key Objectives:

* **Predict Late Delivery Risk:** Build a robust, leakage-free pipeline to flag high-risk orders at or shortly after order approval (`order_approved_at`) before shipping begins.
* **Identify Operational Risk Drivers:** Utilize SHAP (SHapley Additive exPlanations) values to identify the primary behavioral features influencing late delivery.
* **Financial Impact Modeling:** Quantify operational losses under Baseline ("Do Nothing") vs. Model Intervention scenarios using empirical FP:FN cost ratios
* **Optimize Business Threshold:** Tune the classification decision threshold to maximize late order capture while also maintaining a precision guardrail to prevent seller alert fatigue.

## 2. Data Sources

* **Primary Dataset (Olist Brazilian E-Commerce Public Dataset):** Anonymized relational database covering ~100,000 orders placed between October 2016 and August 2018 across Brazilian sales channels, comprising 9 core relational CSV files:
* * `olist_orders_dataset.csv`: Core order lifecycle timestamps (`purchase`,`approved`,`carrier_delivered`,`customer_delivered`,`estimated_delivery`) and order statuses.
  * `olist_order_items_dataset.csv`: Item details, price, freight charges, seller associations, and item volume per order.
  * `olist_order_payments_dataset.csv`: Transaction methods (`boleto`,`credit_card`,`debit_card`,`voucher`), installment structures, and total values.
  * `olist_order_reviews_dataset.csv`: Customer satisfaction feedback ratings (1 to 5 stars) and review comment with submission dates.
  * `olist_products_dataset.csv`: Product physical attributes (weight, length, height, width) and original Portuguese category names.
  * `product_category_name_translation.csv`: Category mapping table converting Portuguese category names to English (`product_category_name_english`).
  * `olist_sellers_dataset.csv`: Merchant locations (ZIP code prefix `seller_zip_code_prefix`, city, state).
  * `olist_customers_dataset.csv`: Buyer locations (ZIP code prefix `customer_zip_code_prefix`, city, state) and customer identifier mappings.
  * `olist_geolocation_dataset.csv`: Brazilian ZIP code centroids (CEP to latitude/longitude coordinates) used for seller-to-customer Haversine transit distance calculations.

## 3. Technologies Used

* **Core Runtime & Environment:** Python 3.12, Jupyter Notebook
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Data Preprocessing & Feature Engineering:**

  * **Scalers & Imputers:** `scikit-learn` (`RobustScaler`, `SimpleImputer`)
  * **Encoders:** `scikit-learn` (`OneHotEncoder`), `category_encoders` (`BinaryEncoder`)
  * **Feature Selection & Custom Pipelines:** `scikit-learn` (`SelectKBest`, `f_classif`, `FunctionTransformer`, `ColumnTransformer`)
  * **Custom Serving Utilities (`Utils.serving_utils`):** Haversine transit distance calculation (`calculate_haversine`), Black Friday and holiday flags (`get_black_friday_date`, `is_black_friday_or_holiday`), historical seller late rate lookups (`attach_seller_late_rate_frozen`), and order-level feature aggregation (`OrderFeatureAggregator`, `aggregate_order_features`).
* **Class Imbalance Handling & Resampling:** `imbalanced-learn` (`ImbPipeline`, `SMOTE`, `SMOTEENN`, `SMOTETomek`)
* **Machine Learning Frameworks & Algorithms Evaluated:**

  * **Tree-Based Ensembles & Boosting:** XGBoost (`XGBClassifier`), LightGBM (`LGBMClassifier`), Random Forest (`RandomForestClassifier`), Extra Trees (`ExtraTreesClassifier`), Gradient Boosting (`GradientBoostingClassifier`), AdaBoost (`AdaBoostClassifier`), Decision Trees (`DecisionTreeClassifier`), Bagging (`BaggingClassifier`)
  * **Linear, SVM & Distance Models:** Logistic Regression (`LogisticRegression`), Ridge (`RidgeClassifier`), SGD (`SGDClassifier`), Linear Support Vector Classifier (`LinearSVC`), Support Vector Classifier (`SVC`)
* **Model Calibration & Probability Alignment:**

  * `scikit-learn` (`CalibratedClassifierCV`, `FrozenEstimator`)
* **Model Validation, Hyperparameter Tuning & Metrics:**

  * **Validation & Search:** `scikit-learn` (`Pipeline`, `TimeSeriesSplit`, `GridSearchCV`, `RandomizedSearchCV`, `cross_validate`, `learning_curve`)
  * **Evaluation Metrics & Curves:** `scikit-learn` (`average_precision_score`, `precision_recall_curve`, `confusion_matrix`, `ConfusionMatrixDisplay`)
* **Model Explainability & Production Serving Wrappers:**

  * **Explainability:** SHAP (`shap`)
  * **Serialization & Production Serving Service:** `pickle`, custom inference wrapper (`ServingModel`, `load_serving_model`, `build_seller_lookup`, `refresh_seller_lookup`)

## 4. Project Structure

```
├── README.md                    <- Top-level README outlining repository, business context, and operational results.
│
├── Assets / Model               <- Serialized model pipelines (.pkl/.joblib)
│
├── Dataset
│   ├── Cleaned Data             <- Cleaned relational tables post-imputation, filtering, and data type casting.
│   │
│   ├── Merged Data              <- Consolidated item-level dataset combining orders, items, payments, sellers, and customer details.
│   │
│   ├── Processed Data           <- seller late-rate lookup table.
│   │
│   ├── Raw Data                 <- Original Olist relational CSV datasets (orders, items, payments, customers, sellers, products, reviews).
│   │
│   └── Supporting Data          <- External geolocation reference data for imputing missing Brazilian ZIP codes (CEP), cities, states, and coordinates.
│
├── notebooks                
│   ├── 1_Data Preparation       <- Data merging, cleaning, and missing imputation.
│   │
│   ├── 2_Business Understanding <- Problem framing, financial cost asymmetry modeling (FP vs. FN), and metric strategy selection.
│   │
│   ├── 3_Exploratory Data Analysis <- Statistical hypothesis testing and delay driver analysis.
│   │
│   ├── 4_Machine Learning Modeling <- Model training, time-series validation, post-training threshold tuning.
│   │
│   └── 5_Recommendation         <- Financial savings synthesis, operational alert strategy, and business implementation plans.
│
├── Utils                        <- Custom Python modules (`serving_utils.py`) containing Haversine formulas, seller lookups, and serving wrappers.
│
└── requirements.txt             <- Dependencies file for reproducing the Python execution environment.
```

## 5. Summary of Findings

### 5.1 Business Insights

* **Model Performance & Screening Function:** The tuned **LightGBM (SMOTE)** model operates at a calibrated threshold of  **0.076** , achieving  **87% Recall** ,  **13% Precision** , an  **F1-score of 0.230** , and a  **PR-AUC of 0.242** . Rather than serving as a rigid final classifier, the model acts as an effective high-sensitivity screening tool—flagging 9,817 out of 19,110 validation orders (531 True Positives) to capture the vast majority of late risks early in the fulfillment cycle.
* **Financial Impact & Customer Satisfaction Recovery:** Implementing proactive intervention on flagged high-risk orders generates  **R$15,793.24 in Net Value** . Crucially, managing customer expectations and mitigating fulfillment delays boosts average customer review scores for impacted orders by  **32.26%** , increasing satisfaction ratings from  **3.01 to 3.98 stars** .
* **Key Risk Drivers (SHAP Analysis):**

  * **Promised Days:** Tightly constrained promised delivery windows represent the single strongest driver pushing predictions toward late status.
  * **Historical Seller Late Rate:** Sellers with past dispatch delays consistently drive higher risk for new orders.
  * **Total Freight Value:** Higher freight costs serve as a proxy for heavier or bulkier items, which suffer from greater handling complexity and transit delays.
  * **Seasonal Vulnerabilities:** Orders placed in **February** (coinciding with Brazil's National Carnival road closures) and **November** (Black Friday logistics volume surges) exhibit significantly higher late delivery rates compared to mid-year months (June–August).

### 5.2 Actionable Recommendations

**Proactive Operational Workflow**

* Flag high-risk order: Use the predictive model to identify orders with a high probability of late delivers.
* Notify sellers: Send an automated notification to the seller and request immediate action to prevent further delays.

* Track shipment & contact logistics partners: Monitor the flagged shipment and coordinate with the logistics partner when the delivery is at risk of exceeding the promised date
* Adjust delivery expectations when necessary: When a delay cannot be prevented, update the estimated delivery timeline to provide a more realistic expectation and reduce unexpected customer dissatisfaction. This is done by adjusting the estimated delivery days with adding extra days to the estimated delivery day based on distance.

| **Transit Distance Category**     | **Added SLA Days** |
| --------------------------------------- | ------------------------ |
| **Short Range (<50 km)**          | +4 Days                  |
| **Regional (50–300 km)**         | +5 Days                  |
| **Interregional (300–1,000 km)** | +7 Days                  |
| **Long-Haul (>1,000 km)**         | +8 Days                  |



**Future Model Enhancements**

* **Traffic and Seasonal Road Disruption Data:** Incorporate seasonal calendar indicators and localized road closure data to account for predictable events like Brazil's Carnival street closures.
* **Real-Time Weather Forecast Data:** Integrate real-time weather forecast API feeds captured at the exact timestamp of order purchase to account for regional storm or flooding delays.
* **Infrastructure and Terrain Profiling:** Feature-engineer road quality indexes, urban/rural destination classifications, and elevation metrics to better distinguish accessible transit routes from difficult terrains.
* **Logistics Partner and Carrier Metrics:** Establish data-sharing agreements with third-party logistics partners to integrate real-time carrier performance metrics and vehicle capacity constraints into the feature pipeline.

# Appendix

Visit our dashboard at ([public.tableau.com/views/Olist-Dashboard_17882680567370/Overview?:language=en-US&amp;publish=yes&amp;:sid=&amp;:redirect=auth&amp;:display_count=n&amp;:origin=viz_share_link](https://public.tableau.com/views/Olist-Dashboard_17882680567370/Overview?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link))
