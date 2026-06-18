import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ==========================
# CẤU HÌNH TRANG
# ==========================
st.set_page_config(
    page_title="SVM Explorer",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 TÌM HIỂU THUẬT TOÁN SVM VÀ ỨNG DỤNG")

# ==========================
# LOAD DATA
# ==========================
data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

# ==========================
# TRAIN MODEL
# ==========================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(
    kernel="rbf",
    probability=True
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

# ==========================
# TABS
# ==========================
tab1, tab2, tab3 = st.tabs([
    "📥 Input dữ liệu",
    "📊 Trực quan hóa",
    "📋 10 dòng dữ liệu đầu"
])

# =====================================================
# TAB 1
# =====================================================
with tab1:

    st.header("📥 Nhập dữ liệu để dự đoán")

    col1, col2 = st.columns(2)

    with col1:
        radius_mean = st.number_input(
            "Radius Mean",
            value=float(df["mean radius"].mean())
        )

        texture_mean = st.number_input(
            "Texture Mean",
            value=float(df["mean texture"].mean())
        )

        perimeter_mean = st.number_input(
            "Perimeter Mean",
            value=float(df["mean perimeter"].mean())
        )

    with col2:
        area_mean = st.number_input(
            "Area Mean",
            value=float(df["mean area"].mean())
        )

        smoothness_mean = st.number_input(
            "Smoothness Mean",
            value=float(df["mean smoothness"].mean())
        )

        compactness_mean = st.number_input(
            "Compactness Mean",
            value=float(df["mean compactness"].mean())
        )

    st.info(
        "Demo nhập 6 thuộc tính chính. "
        "Các thuộc tính còn lại sẽ được gán giá trị trung bình."
    )

    if st.button("🚀 Dự đoán"):

        sample = X.mean().values

        feature_names = list(X.columns)

        sample[feature_names.index("mean radius")] = radius_mean
        sample[feature_names.index("mean texture")] = texture_mean
        sample[feature_names.index("mean perimeter")] = perimeter_mean
        sample[feature_names.index("mean area")] = area_mean
        sample[feature_names.index("mean smoothness")] = smoothness_mean
        sample[feature_names.index("mean compactness")] = compactness_mean

        sample = sample.reshape(1, -1)

        sample_scaled = scaler.transform(sample)

        prediction = model.predict(sample_scaled)[0]
        probability = model.predict_proba(sample_scaled).max()

        st.subheader("Kết quả dự đoán")

        if prediction == 1:
            st.success("✅ Khối u lành tính (Benign)")
        else:
            st.error("⚠️ Khối u ác tính (Malignant)")

        st.write(f"Độ tin cậy: {probability*100:.2f}%")

        st.write(f"Độ chính xác mô hình: {accuracy*100:.2f}%")

# =====================================================
# TAB 2
# =====================================================
with tab2:

    st.header("📊 Trực quan hóa dữ liệu")

    # ===== 1 =====
    st.subheader("1. Phân bố nhãn")

    fig, ax = plt.subplots()

    sns.countplot(
        x="target",
        data=df,
        ax=ax
    )

    ax.set_title("Class Distribution")

    st.pyplot(fig)

    # ===== 2 =====
    st.subheader("2. Histogram Mean Radius")

    fig, ax = plt.subplots()

    sns.histplot(
        df["mean radius"],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

    # ===== 3 =====
    st.subheader("3. Boxplot Mean Texture")

    fig, ax = plt.subplots()

    sns.boxplot(
        x=df["mean texture"],
        ax=ax
    )

    st.pyplot(fig)

    # ===== 4 =====
    st.subheader("4. Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(12, 8))

    corr = df.corr()

    sns.heatmap(
        corr,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # ===== 5 =====
    st.subheader("5. Scatter Plot")

    fig, ax = plt.subplots()

    sns.scatterplot(
        data=df,
        x="mean radius",
        y="mean texture",
        hue="target",
        ax=ax
    )

    st.pyplot(fig)

# =====================================================
# TAB 3
# =====================================================
with tab3:

    st.header("📋 10 dòng dữ liệu đầu tiên")

    st.dataframe(df.head(10))

    st.subheader("Thông tin dữ liệu")

    st.write(f"Số dòng: {df.shape[0]}")
    st.write(f"Số cột: {df.shape[1]}")

    st.write("Thống kê mô tả:")

    st.dataframe(df.describe())