# User Journey Analysis
Machine Learning Pipeline and Real-Time Conversion Prediction

This project involved the design and implementation of an end-to-end machine learning pipeline that processes multi-session user data for conversion prediction. The pipeline includes modular components for data ingestion, transformation, and model training, ensuring efficient handling of large datasets and high-quality predictions.

Key Features:

Data Preprocessing: Built a robust pipeline to clean and transform user session data, handle duplicate records, and label conversion outcomes based on checkout behavior.

Model Training: Trained multiple classification models, including Logistic Regression, Random Forest, and XGBoost, using scikit-learn. Achieved a ROC-AUC score of 0.84 for user-level conversion prediction, demonstrating high model accuracy.

Real-Time Prediction Service: Deployed a Flask-based REST API on AWS Elastic Beanstalk, providing a scalable solution for serving real-time conversion probability predictions for new user journeys.

This project showcases the application of machine learning techniques in production environments, offering accurate, on-demand predictions to optimize business decisions.
