import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="SVM Explorer",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>

.main-title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:#2563eb;
}

.sub-title{
    text-align:center;
    color:#6b7280;
    font-size:18px;
    margin-bottom:25px;
}

.block-container{
    padding-top:1rem;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
data = load_breast_cancer()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["target"] = data.target

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.title("⚙️ Model Settings")

    kernel = st.selectbox(
        "Kernel",
        ["linear", "rbf", "poly", "sigmoid"]
    )

    st.markdown("---")

    st.info("""
    Support Vector Machine (SVM)

    Ứng dụng:
    • Phân loại dữ liệu
    • Dự đoán bệnh
    • Nhận diện mẫu
    • Machine Learning
    """)

# =====================================================
# TRAIN MODEL
# =====================================================
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
    kernel=kernel,
    probability=True
)

model.fit(
    X_train_scaled,
    y_train
)

pred = model.predict(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    pred
)

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="main-title">
🤖 SVM Explorer
</div>

<div class="sub-title">
Breast Cancer Prediction using Support Vector Machine
</div>
""", unsafe_allow_html=True)

# =====================================================
# KPI DASHBOARD
# =====================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "📊 Samples",
    df.shape[0]
)

c2.metric(
    "📋 Features",
    X.shape[1]
)

c3.metric(
    "🎯 Accuracy",
    f"{accuracy*100:.2f}%"
)

c4.metric(
    "🧠 Kernel",
    kernel.upper()
)

st.markdown("---")

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(
[
    "📥 Input dữ liệu",
    "📊 Trực quan hóa",
    "📋 Dataset"
]
)

# =====================================================
# TAB 1
# =====================================================
with tab1:

    st.subheader("⚙️ Cấu hình và huấn luyện SVM")

    st.markdown("""
    Support Vector Machine (SVM) là thuật toán học máy dùng cho
    các bài toán phân loại và hồi quy.

    Người dùng có thể thay đổi các tham số để quan sát ảnh hưởng
    đến hiệu năng mô hình.
    """)

    # ==========================
    # CẤU HÌNH
    # ==========================

    col1, col2 = st.columns(2)

    with col1:

        kernel = st.selectbox(
            "Kernel Function",
            ["linear", "rbf", "poly", "sigmoid"]
        )

        C = st.slider(
            "Regularization Parameter (C)",
            0.1,
            100.0,
            1.0
        )

    with col2:

        gamma = st.selectbox(
            "Gamma",
            ["scale", "auto"]
        )

        test_size = st.slider(
            "Test Size",
            0.1,
            0.5,
            0.2
        )

    degree = 3

    if kernel == "poly":

        degree = st.slider(
            "Polynomial Degree",
            2,
            10,
            3
        )

    # ==========================
    # TRAIN MODEL
    # ==========================

    if st.button(
        "🚀 Train Model",
        use_container_width=True
    ):

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            degree=degree,
            probability=True
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        y_pred = model.predict(
            X_test_scaled
        )

        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred
        )

        recall = recall_score(
            y_test,
            y_pred
        )

        f1 = f1_score(
            y_test,
            y_pred
        )

        st.success("Huấn luyện thành công!")

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric(
            "Accuracy",
            f"{accuracy*100:.2f}%"
        )

        m2.metric(
            "Precision",
            f"{precision*100:.2f}%"
        )

        m3.metric(
            "Recall",
            f"{recall*100:.2f}%"
        )

        m4.metric(
            "F1 Score",
            f"{f1*100:.2f}%"
        )

        m5.metric(
            "Support Vectors",
            int(sum(model.n_support_))
        )

        st.markdown("---")

        st.subheader("📌 Thông tin mô hình")

        st.write(f"Kernel: {kernel}")
        st.write(f"C: {C}")
        st.write(f"Gamma: {gamma}")
        st.write(f"Support Vectors: {sum(model.n_support_)}")

        st.markdown("---")

        st.subheader("🎯 Demo dự đoán")

        c1, c2, c3 = st.columns(3)

        with c1:
            radius = st.number_input(
                "Radius Mean",
                value=float(df["mean radius"].mean())
            )

        with c2:
            texture = st.number_input(
                "Texture Mean",
                value=float(df["mean texture"].mean())
            )

        with c3:
            perimeter = st.number_input(
                "Perimeter Mean",
                value=float(df["mean perimeter"].mean())
            )

        if st.button(
            "Predict Sample"
        ):

            sample = X.mean().values

            cols = list(X.columns)

            sample[
                cols.index("mean radius")
            ] = radius

            sample[
                cols.index("mean texture")
            ] = texture

            sample[
                cols.index("mean perimeter")
            ] = perimeter

            sample = sample.reshape(
                1,
                -1
            )

            sample_scaled = scaler.transform(
                sample
            )

            pred = model.predict(
                sample_scaled
            )[0]

            prob = model.predict_proba(
                sample_scaled
            ).max()

            if pred == 1:

                st.success(
                    f"✅ Benign ({prob*100:.2f}%)"
                )

            else:

                st.error(
                    f"⚠️ Malignant ({prob*100:.2f}%)"
                )
# =====================================================
# TAB 2
# =====================================================
with tab2:

    st.subheader("📊 Trực quan hóa dữ liệu")

    # 1
    st.markdown("### 1. Class Distribution")

    fig = px.histogram(
        df,
        x="target",
        color="target"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # 2
    st.markdown("### 2. Histogram - Mean Radius")

    fig = px.histogram(
        df,
        x="mean radius"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # 3
    st.markdown("### 3. Boxplot - Mean Texture")

    fig = px.box(
        df,
        y="mean texture"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # 4
    st.markdown("### 4. Correlation Heatmap")

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    sns.heatmap(
        df.corr(),
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

    # 5
    st.markdown("### 5. Scatter Plot")

    fig = px.scatter(
        df,
        x="mean radius",
        y="mean texture",
        color="target"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# TAB 3
# =====================================================
with tab3:

    st.subheader("📋 10 dòng dữ liệu đầu tiên")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("### 📈 Thống kê dữ liệu")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        "⬇ Download Dataset",
        csv,
        "breast_cancer_dataset.csv",
        "text/csv"
    )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")

st.markdown("""
<div style='text-align:center;color:gray;'>

Support Vector Machine Project

Nhập môn Khoa học dữ liệu

Huế University

</div>
""", unsafe_allow_html=True)
