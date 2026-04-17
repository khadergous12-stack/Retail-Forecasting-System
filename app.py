import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from src.data_preprocessing import load_data
from src.feature_engineering import create_features
from src.model import train_model
from src.inventory import inventory_policy

sns.set_theme(style="whitegrid")

# ================= PAGE =================
st.set_page_config(page_title="Retail Forecasting System", layout="wide")

# ================= LOAD =================
@st.cache_data
def load():
    return load_data()

df = load()

# ================= SIDEBAR =================
st.sidebar.header("Controls")

store = st.sidebar.selectbox("Store", sorted(df["store_id"].unique()))
item = st.sidebar.selectbox("Item", sorted(df["item_id"].unique()))
lead_time = st.sidebar.slider("Lead Time", 1, 14, 7)
stock = st.sidebar.slider("Current Stock", 0, 50000, 200)

# ================= FILTER =================
df = df[(df["store_id"] == store) & (df["item_id"] == item)]
df = create_features(df)

# ================= MODEL =================
model, preds, actual, feature_names = train_model(df)

# ================= INVENTORY =================
inv = inventory_policy(preds, lead_time=lead_time, on_hand=stock)

# ================= TITLE =================
st.title("Retail Forecasting System")

# ================= KPI =================
st.markdown("### 🟢 Inventory Insights")

c1, c2, c3 = st.columns(3)
c1.metric("Safety Stock", int(inv["Safety Stock"]))
c2.metric("Reorder Point", int(inv["Reorder Point"]))
c3.metric("Order Quantity", int(inv["Order Quantity"]))

# ================= GRAPH SETTINGS =================
n = min(60, len(actual), len(preds))

def create_fig():
    fig, ax = plt.subplots(figsize=(6,3))
    return fig, ax

# ================= GRAPH ROW 1 =================
st.markdown("### 🔵 Forecast Analysis")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = create_fig()
    ax1.plot(actual.values[:n], color="blue", label="Actual")
    ax1.plot(preds[:n], color="orange", linestyle="--", label="Forecast")
    ax1.legend()
    ax1.grid(alpha=0.3)
    st.pyplot(fig1)

with col2:
    trend = df.set_index("date")["qty_sold"].rolling(7).mean()
    fig2, ax2 = create_fig()
    ax2.plot(trend, color="green", label="Trend")

    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30)

    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

# ================= GRAPH ROW 2 =================
st.markdown("### 🟣 Distribution & Errors")

col3, col4 = st.columns(2)

with col3:
    fig3, ax3 = create_fig()
    sns.histplot(df["qty_sold"], kde=True, color="purple", ax=ax3)
    st.pyplot(fig3)

with col4:
    errors = actual.values[:n] - preds[:n]
    fig4, ax4 = create_fig()
    ax4.plot(errors, color="red", label="Error")
    ax4.axhline(0, linestyle="--", color="black")
    ax4.legend()
    st.pyplot(fig4)

# ================= FEATURE =================
st.markdown("### 🟠 Feature Importance")

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance")

fig5, ax5 = plt.subplots(figsize=(8,4))
sns.barplot(data=feat_df, x="Importance", y="Feature", color="teal", ax=ax5)
st.pyplot(fig5)

# ================= DATA PREVIEW (FIXED BACK) =================
st.markdown("### 📊 Data Preview")
st.dataframe(df.tail(20))

# ================= PDF =================
st.markdown("### 📄 Download Full Report")

def save_plot(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf

def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    elements = []

    # TITLE
    elements.append(Paragraph("<font color='blue'><b>Retail Forecasting System Report</b></font>", styles["Title"]))
    elements.append(Spacer(1, 12))

    # KPI
    elements.append(Paragraph("<font color='green'><b>Inventory Insights</b></font>", styles["Heading2"]))

    kpi_table = Table([
        ["Metric", "Value"],
        ["Safety Stock", int(inv["Safety Stock"])],
        ["Reorder Point", int(inv["Reorder Point"])],
        ["Order Quantity", int(inv["Order Quantity"])]
    ])

    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER")
    ]))

    elements.append(kpi_table)
    elements.append(Spacer(1, 12))

    # GRAPHS
    elements.append(Paragraph("<font color='blue'><b>Forecast vs Actual</b></font>", styles["Heading2"]))
    elements.append(Image(save_plot(fig1), width=400, height=200))

    elements.append(Paragraph("<font color='green'><b>Sales Trend</b></font>", styles["Heading2"]))
    elements.append(Image(save_plot(fig2), width=400, height=200))

    elements.append(Paragraph("<font color='purple'><b>Distribution</b></font>", styles["Heading2"]))
    elements.append(Image(save_plot(fig3), width=400, height=200))

    elements.append(Paragraph("<font color='red'><b>Residuals</b></font>", styles["Heading2"]))
    elements.append(Image(save_plot(fig4), width=400, height=200))

    elements.append(Paragraph("<font color='orange'><b>Feature Importance</b></font>", styles["Heading2"]))
    elements.append(Image(save_plot(fig5), width=400, height=200))

    doc.build(elements)
    buffer.seek(0)
    return buffer

pdf = generate_pdf()

st.download_button(
    label="⬇️ Download Full Report (PDF)",
    data=pdf,
    file_name="Retail_Forecasting_Report.pdf",
    mime="application/pdf"
)