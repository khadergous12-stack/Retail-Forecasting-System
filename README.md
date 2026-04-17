# 📊 Retail Forecasting System

A complete end-to-end machine learning project that predicts retail product demand and optimizes inventory decisions using data-driven insights. This system integrates forecasting, visualization, API access, and automated reporting into a single unified solution.

---

## 📌 Project Overview

In retail businesses, incorrect demand prediction can lead to stock shortages or excess inventory. This project solves that problem by building a predictive system that forecasts future sales and converts those predictions into actionable inventory decisions such as safety stock, reorder point, and order quantity.

The system provides both an interactive dashboard and API endpoints, making it suitable for real-world integration scenarios.

---

## 🎯 Objectives

- Predict future product demand using historical sales data  
- Optimize inventory to avoid stockouts and overstocking  
- Visualize trends, patterns, and model performance  
- Provide API-based access for integration  
- Generate downloadable reports for business use  

---

## 🛠️ Technologies Used

- **Python**  
- **Pandas, NumPy** (Data Processing)  
- **Matplotlib, Seaborn** (Visualization)  
- **Scikit-learn (Random Forest)** (Machine Learning)  
- **Streamlit** (Interactive Dashboard)  
- **FastAPI** (Backend API)  
- **ReportLab** (PDF Report Generation)  

---

## 📂 Project Structure

```
Retail-Forecasting-System/
│
├── data/
│   ├── train.csv
│   ├── features.csv
│   └── stores.csv
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── inventory.py
│
├── app.py        # Streamlit Dashboard
├── api.py        # FastAPI Backend
├── requirements.txt
└── README.md
```

---

## ⚙️ System Workflow

1. Data Loading & Cleaning – Load and preprocess datasets  
2. Feature Engineering – Create lag features, rolling averages, and date features  
3. Model Training – Train Random Forest model for prediction  
4. Prediction – Generate future demand values  
5. Evaluation – Calculate MAE, RMSE, and MASE  
6. Inventory Optimization – Compute safety stock, reorder point, and order quantity  
7. Visualization – Display insights through graphs  
8. API Exposure – Provide endpoints for prediction and inventory  
9. Report Generation – Generate downloadable PDF report  

---

## 📊 Dashboard Features

- 📈 Forecast vs Actual Comparison  
- 📊 Sales Trend Analysis  
- 📦 Sales Distribution  
- ⚠️ Residual (Error) Analysis  
- 🔥 Feature Importance  
- 📦 Inventory Metrics:
  - Safety Stock  
  - Reorder Point  
  - Order Quantity  

---

## 🌐 API Endpoints

The project also exposes REST APIs using FastAPI:

- `/` → Health check  
- `/forecast` → Returns sample forecast values  
- `/inventory` → Returns inventory recommendations  
- `/metrics` → Returns MAE and RMSE  

These endpoints allow integration with external systems or applications.

---

## 📄 Report Generation

The system includes a PDF download feature that provides:

- Inventory KPI table  
- Forecast vs Actual graph  
- Sales trend visualization  
- Distribution and residual analysis  
- Feature importance chart  

This report helps in real-world business decision-making.

---

## 🧪 Virtual Simulation

This project simulates real-world retail operations using historical datasets instead of live production data.

- Seasonal trends are captured using time-based features  
- Demand variability is modeled using lag and rolling features  
- Inventory decisions are derived using predicted demand and lead-time assumptions  

This simulation approach allows testing and validation of forecasting and inventory strategies in a controlled environment.

---

## ▶️ How to Run

### Run Dashboard

```bash
streamlit run app.py
```

### Run API

```bash
uvicorn api:app --reload
```

Open:
```
http://127.0.0.1:8000/docs
```

---

## 📈 Output

- Interactive dashboard with filters  
- Accurate demand predictions  
- Inventory recommendations  
- Downloadable PDF report  
- API-based access to predictions  

---

## 🧠 Key Insights

- Demand shows seasonal trends  
- Lag features significantly improve prediction accuracy  
- Rolling averages stabilize predictions  
- Inventory optimization reduces stock risk  

---

## 🚀 Future Enhancements

- Multi-store comparison dashboard  
- Real-time forecasting system  
- Deployment on cloud (Streamlit Cloud / AWS)  
- Advanced models like XGBoost or LSTM  

---

## 👨‍💻 Author

Khader Gouse  

---

## ⭐ Conclusion

This project demonstrates how machine learning can be applied to real-world retail problems by combining forecasting, analytics, API integration, and reporting into one unified system for smarter inventory management.
