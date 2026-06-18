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

    st.subheader("📥 Nhập thông tin bệnh nhân")

    col1, col2, col3 = st.columns(3)

    with col1:
        radius = st.number_input(
            "Radius Mean",
            value=float(df["mean radius"].mean())
        )

        area = st.number_input(
            "Area Mean",
            value=float(df["mean area"].mean())
        )

    with col2:
        texture = st.number_input(
            "Texture Mean",
            value=float(df["mean texture"].mean())
        )

        smoothness = st.number_input(
            "Smoothness Mean",
            value=float(df["mean smoothness"].mean())
        )

    with col3:
        perimeter = st.number_input(
            "Perimeter Mean",
            value=float(df["mean perimeter"].mean())
        )

        compactness = st.number_input(
            "Compactness Mean",
            value=float(df["mean compactness"].mean())
        )

    st.info(
        "Các thuộc tính còn lại được gán bằng giá trị trung bình của tập dữ liệu."
    )

    if st.button(
        "🚀 Predict",
        use_container_width=True
    ):

        sample = X.mean().values

        columns = list(X.columns)

        sample[columns.index("mean radius")] = radius
        sample[columns.index("mean texture")] = texture
        sample[columns.index("mean perimeter")] = perimeter
        sample[columns.index("mean area")] = area
        sample[columns.index("mean smoothness")] = smoothness
        sample[columns.index("mean compactness")] = compactness

        sample = sample.reshape(1, -1)

        sample_scaled = scaler.transform(sample)

        prediction = model.predict(
            sample_scaled
        )[0]

        probability = model.predict_proba(
            sample_scaled
        ).max()

        st.markdown("### 🎯 Kết quả dự đoán")

        if prediction == 1:

            st.success(
                f"""
                ✅ Benign (Lành tính)

                Độ tin cậy: {probability*100:.2f}%
                """
            )

        else:

            st.error(
                f"""
                ⚠️ Malignant (Ác tính)

                Độ tin cậy: {probability*100:.2f}%
                """
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
