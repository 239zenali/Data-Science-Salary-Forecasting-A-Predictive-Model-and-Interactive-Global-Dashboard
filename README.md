__📊 Data Science Salary Forecasting
Predictive Model & Interactive Global Dashboard__


📌 Project Overview

This project presents a complete end-to-end Data Science Salary Forecasting System that combines:

📈 Machine Learning-based salary prediction

🏢 Company recommendation engine

🌍 Interactive global dashboard

📊 Data visualization using Power BI

The system enables users to:
Predict expected salary based on job profile
Explore global salary trends
Identify high-paying companies
Analyze salary distribution across countries, experience levels, and job roles

🎯 Problem Statement

Despite abundant salary data from job portals and surveys, there is no unified system that:
Predicts salaries based on individual profiles
Provides company recommendations
Offers interactive global visualization
This project solves that gap by integrating predictive modeling with real-time dashboards.

📂 Datasets Used

Global AI/ML & Data Science Salary Dataset (Kaggle)
Company-level Dataset for Recommendation

These datasets include:
Job Title
Experience Level
Employment Type
Company Size
Remote Work Ratio
Employee Residence
Company Location
Salary (Converted to USD)

⚙️ Technologies Used

🔹 Programming & ML
Python
Pandas
NumPy
Scikit-learn
XGBoost
Random Forest
LightGBM (Final Model)

🔹 Visualization

Matplotlib
Seaborn
Power BI

🔹 Deployment

Streamlit
Google Colab

🤖 Machine Learning Models Compared

Linear Regression
Ridge & Lasso Regression
Decision Tree Regressor
Random Forest Regressor
KNN Regressor
SVR
Gradient Boosting
XGBoost
LightGBM (Best Performing Model)

📌 Final Model: LightGBM

LightGBM demonstrated superior performance in:

R² Score
RMSE
MAE

It was selected as the final deployed model.

🔮 Prediction Logic

User inputs:
Job Title
Experience Level
Employment Type
Employee Residence
Company Location
Remote Work Ratio
Company Size

Input preprocessing:

One-hot encoding
Feature alignment
Scaling consistency

Model inference:

salary = model.predict(user_input)

Salary displayed in USD (rounded & formatted)

🏢 Company Recommendation Algorithm

After predicting salary:

Filter companies by:
Job role
Country
Compute average company salary:
company_means = df.groupby('company')['salary'].mean()


Select companies where:

avg_salary >= predicted_salary


Rank and display Top 5 companies.

📊 Streamlit Dashboard

The Streamlit web app provides:

🔹 Inputs:

Job Role

Experience Level
Employment Type
Location
Remote Ratio
Company Size

🔹 Outputs:

💰 Predicted Salary
🏆 Top 5 Recommended Companies
It enables real-time interaction and smooth inference.

📈 Power BI Dashboard Features

The Power BI dashboard provides interactive global insights (2020–2025):

Filters Available:
Company Location
Experience Level
Employment Type
Company Size
Remote Ratio
Work Year
Job Title

Visual Insights:

🌍 Country-wise salary map
📊 Salary by experience level
🏢 Salary by company size
📅 Salary trend over time
💼 Top-paying job titles
🌐 Global salary distribution

📊 Key Analytical Insights
💼 Salary by Employment Type

Full-time roles offer highest average salaries
Freelancers & part-time earn comparatively less

📈 Salary by Experience

Strong positive correlation between experience and salary
Senior roles command premium compensation

🌍 Global Trends

USA & Israel lead in salary averages
North America & Western Europe dominate high salary zones
Developing regions show lower compensation levels

🏢 Company Size Impact

Medium-sized companies offer highest average pay
Small companies show higher variability

🏠 Remote Work Impact

Fully remote roles offer competitive compensation
Remote work does not reduce salary significantly

📌 Applications

🎓 Career Planning for Students
💼 Salary Negotiation Support
🏢 HR Salary Benchmarking
🌍 Global Market Trend Analysis
📊 Workforce Planning & Policy Insights


🔮 Future Enhancements

Deep learning models for improved prediction
Skill-based salary modeling
Real-time API data integration
Enhanced company recommendation scoring
Multilingual dashboard support
Full-scale web deployment

