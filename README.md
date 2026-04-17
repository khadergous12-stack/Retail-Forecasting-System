# 📊 Retail Forecasting System

A complete end-to-end machine learning project that predicts retail product demand and optimizes inventory decisions using data-driven insights. This system integrates forecasting, visualization, and automated reporting into a single interactive dashboard.

---

## 📌 Project Overview

In retail businesses, incorrect demand prediction can lead to stock shortages or excess inventory. This project solves that problem by building a predictive system that forecasts future sales and converts those predictions into actionable inventory decisions such as safety stock, reorder point, and order quantity.

The system also provides visual insights and generates structured reports to support business decision-making.

---

## 🎯 Objectives

- Predict future product demand using historical sales data  
- Optimize inventory to avoid stockouts and overstocking  
- Visualize trends, patterns, and model performance  
- Generate downloadable reports for business use  

---

## 🛠️ Technologies Used

- **Python**  
- **Pandas, NumPy** (Data Processing)  
- **Matplotlib, Seaborn** (Visualization)  
- **Scikit-learn (Random Forest)** (Machine Learning)  
- **Streamlit** (Interactive Dashboard)  
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
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ System Workflow

1. Data Loading & Cleaning – Load and preprocess datasets  
2. Feature Engineering – Create lag features, rolling averages, and date features  
3. Model Training – Train Random Forest model for prediction  
4. Prediction – Generate future demand values  
5. Evaluation – Calculate MAE and RMSE  
6. Inventory Optimization – Compute safety stock, reorder point, and order quantity  
7. Visualization – Display insights through graphs  
8. Report Generation – Generate downloadable PDF report  

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

## 📄 Report Generation

The system includes a PDF download feature that provides:

- Inventory KPI table  
- Forecast vs Actual graph  
- Sales trend visualization  
- Distribution and residual analysis  
- Feature importance chart  

This report helps in real-world business decision-making.

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/Retail-Forecasting-System.git
cd Retail-Forecasting-System
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

---

## 📈 Output

- Interactive dashboard with filters  
- Accurate demand predictions  
- Inventory recommendations  
- Downloadable PDF report  

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

This project demonstrates how machine learning can be applied to real-world retail problems by combining forecasting, analytics, and reporting into one unified system for smarter inventory management.
