**# ai-crop-irrigation-predict-dashboard
**An AI-powered Precision Irrigation Dashboard that predicts crop-specific water requirements using machine learning and a web-based Streamlit UI.

**🌾 Precision Irrigation & Crop Water Requirement (CWR) System
**This repository contains an end-to-end Machine Learning and analytics application designed to optimize agricultural water usage. Using a Random Forest Classifier, the system predicts irrigation intensity requirements ("Low", "Medium", "High") based on environmental parameters, soil types, and crop growth stages. It then converts these predictions into exact numeric water volume recommendations (in Liters) tailored to the user's specific field size.

**🚀 Key Features
**1.Predictive AI Engine: Trains a Random Forest model on historical agricultural data (irrigation_prediction.csv) to classify water requirements.  
2.Dynamic Water Volume Calculation: Converts categorical AI predictions into localized water depth (mm) and integrates crop coefficients 
3.Interactive Dashboard: A clean, user-friendly Streamlit UI allowing farmers or researchers to input real-time field data (soil moisture, temperature, field area,   growth stage) and view instant metrics.  

**🛠️ Tech Stack
**Language: Python
Frameworks & Libraries: Streamlit, Pandas, NumPy, Scikit-learn
Data Serialization: Pickle, JSON

**📁 Core Files Described
**1.irrigation_prediction.csv: The underlying master dataset containing environmental features and historical irrigation targets used to train the classifier.  2.model_trainer.py: The preprocessing and training script that performs one-hot encoding on features, saves the feature alignment pipeline (model_columns.pkl), and exports the trained Random Forest model (crop_model.pkl).  
3.utils.py: Contains core business logic helpers such as data preprocessing steps and the get_water_depth() conversion mapping. 
4.crop_data.json: Stores standard crop coefficients ($K_c$) for agricultural profiles including Rice, Maize, Wheat, and Sugarcane.  
5.app.py: The interactive frontend dashboard built with Streamlit that ties the predictive model and data visualizations together.
