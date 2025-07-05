import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="💎 Diamond Price Analyzer", layout="wide")

st.title("💎 Diamond Price Analysis & Prediction")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("diamond.csv")  # Make sure the file is named correctly
    df.columns = df.columns.str.strip().str.lower()  # Normalize column names
    return df

df = load_data()

# Show raw data and columns
st.subheader("🔍 Preview of Dataset")
st.dataframe(df.head())

st.write("📌 Available Columns:", df.columns.tolist())

# Check required column
if 'price' not in df.columns:
    st.error("❌ Column 'price' not found in your dataset. Please check your CSV file.")
    st.stop()

# Visualizations
st.subheader("📊 Price Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df['price'], kde=True, ax=ax1, color='skyblue')
st.pyplot(fig1)

# Correlation heatmap
st.subheader("🔗 Correlation Heatmap")
fig2, ax2 = plt.subplots()
sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap="coolwarm", ax=ax2)
st.pyplot(fig2)

# Average price by cut (if present)
if 'cut' in df.columns:
    st.subheader("✂️ Average Price by Cut")
    fig3, ax3 = plt.subplots()
    sns.barplot(data=df, x='cut', y='price', ax=ax3)
    st.pyplot(fig3)

# Sidebar: Prediction input
st.sidebar.header("🔮 Predict Diamond Price")

# Helper to handle missing columns
def safe_get(col_name, default=0):
    return st.sidebar.slider(col_name.capitalize(), float(df[col_name].min()), float(df[col_name].max()), float(df[col_name].mean())) if col_name in df.columns else default

# User input
carat = safe_get("carat")
depth = safe_get("depth")
table = safe_get("table")
size = safe_get("size")

# Handle categories safely
cut = st.sidebar.selectbox("Cut", df['cut'].unique()) if 'cut' in df.columns else None
colour = st.sidebar.selectbox("Colour", df['colour'].unique()) if 'colour' in df.columns else None
clarity = st.sidebar.selectbox("Clarity", df['clarity'].unique()) if 'clarity' in df.columns else None

# Prepare features
X = df.drop(['price'], axis=1)

# Encode categorical columns
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col])

y = df['price']

# Model
model = LinearRegression()
model.fit(X, y)

# Prepare input row
input_dict = {
    'carat': carat,
    'depth': depth,
    'table': table,
    'size': size
}

# Add encoded categorical inputs
if cut: input_dict['cut'] = LabelEncoder().fit(df['cut']).transform([cut])[0]
if colour: input_dict['colour'] = LabelEncoder().fit(df['colour']).transform([colour])[0]
if clarity: input_dict['clarity'] = LabelEncoder().fit(df['clarity']).transform([clarity])[0]
