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

    st.header("📊 Trực quan hóa dữ liệu")

    st.markdown("""
    Các biểu đồ dưới đây giúp phân tích dữ liệu trước khi xây dựng mô hình SVM.
    Mỗi biểu đồ thể hiện một góc nhìn khác nhau về tập dữ liệu Breast Cancer.
    """)

    # =====================================================
    # BIỂU ĐỒ 1
    # CLASS DISTRIBUTION
    # =====================================================

    st.subheader("1️⃣ Class Distribution")

    class_counts = (
        df["target"]
        .map({
            0: "Malignant",
            1: "Benign"
        })
        .value_counts()
    )

    fig1 = px.bar(
        x=class_counts.index,
        y=class_counts.values,
        color=class_counts.index,
        text=class_counts.values,
        title="Phân bố số lượng mẫu theo lớp"
    )

    fig1.update_layout(
        xaxis_title="Loại khối u",
        yaxis_title="Số lượng mẫu",
        height=500
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.info("""
    Biểu đồ cho thấy số lượng mẫu Benign (lành tính)
    và Malignant (ác tính) trong tập dữ liệu.
    """)

    st.markdown("---")

    # =====================================================
    # BIỂU ĐỒ 2
    # HISTOGRAM
    # =====================================================

    st.subheader("2️⃣ Distribution of Mean Radius")

    fig2 = px.histogram(
        df,
        x="mean radius",
        nbins=30,
        marginal="box",
        title="Phân bố Mean Radius"
    )

    fig2.update_layout(
        xaxis_title="Mean Radius",
        yaxis_title="Frequency",
        height=500
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.info("""
    Histogram cho biết phân bố kích thước trung bình của khối u.
    Đây là một thuộc tính quan trọng trong chẩn đoán.
    """)

    st.markdown("---")

    # =====================================================
    # BIỂU ĐỒ 3
    # BOXPLOT
    # =====================================================

    st.subheader("3️⃣ Mean Texture Analysis")

    fig3 = px.box(
        df,
        y="mean texture",
        color=df["target"].map({
            0: "Malignant",
            1: "Benign"
        }),
        title="So sánh Mean Texture giữa các lớp"
    )

    fig3.update_layout(
        xaxis_title="Class",
        yaxis_title="Mean Texture",
        height=500
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.info("""
    Boxplot giúp phát hiện các giá trị ngoại lai (Outliers)
    và so sánh phân bố Mean Texture giữa hai nhóm dữ liệu.
    """)

    st.markdown("---")

    # =====================================================
    # BIỂU ĐỒ 4
    # HEATMAP
    # =====================================================

    st.subheader("4️⃣ Correlation Heatmap")

    selected_columns = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "target"
    ]

    corr_matrix = df[selected_columns].corr()

    fig4, ax = plt.subplots(
        figsize=(10, 6)
    )

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="RdBu_r",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Correlation Matrix"
    )

    st.pyplot(fig4)

    st.info("""
    Heatmap thể hiện mức độ tương quan giữa các thuộc tính.
    Giá trị càng gần 1 hoặc -1 thì mối tương quan càng mạnh.
    """)

    st.markdown("---")

    # =====================================================
    # BIỂU ĐỒ 5
    # SCATTER PLOT
    # =====================================================

    st.subheader("5️⃣ Feature Separation")

    fig5 = px.scatter(
        df,
        x="mean radius",
        y="mean texture",
        color=df["target"].map({
            0: "Malignant",
            1: "Benign"
        }),
        size="mean area",
        hover_data=[
            "mean perimeter",
            "mean smoothness"
        ],
        title="Phân tách dữ liệu theo đặc trưng"
    )

    fig5.update_layout(
        xaxis_title="Mean Radius",
        yaxis_title="Mean Texture",
        height=600
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    st.info("""
    Scatter Plot giúp trực quan hóa khả năng phân tách dữ liệu.
    Đây là cơ sở để thuật toán SVM tìm siêu phẳng (Hyperplane)
    tối ưu nhằm phân loại các lớp.
    """)
# =====================================================
# TAB 3
# =====================================================
with tab3:

    st.header("📋 Dataset Explorer")

    st.markdown("""
    Khám phá bộ dữ liệu Breast Cancer Wisconsin Dataset được sử dụng
    trong quá trình huấn luyện và đánh giá mô hình SVM.
    """)

    # =====================================================
    # THÔNG TIN CHUNG
    # =====================================================

    st.subheader("📌 Thông tin tổng quan")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Số dòng",
        df.shape[0]
    )

    c2.metric(
        "Số cột",
        df.shape[1]
    )

    c3.metric(
        "Số đặc trưng",
        X.shape[1]
    )

    c4.metric(
        "Số lớp",
        len(df["target"].unique())
    )

    st.markdown("---")

    # =====================================================
    # TOP 10 DÒNG ĐẦU
    # =====================================================

    st.subheader("📄 10 dòng dữ liệu đầu tiên")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # THỐNG KÊ MÔ TẢ
    # =====================================================

    st.subheader("📈 Thống kê mô tả")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # KIỂM TRA DỮ LIỆU THIẾU
    # =====================================================

    st.subheader("🔍 Kiểm tra dữ liệu thiếu")

    missing_data = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum().values
    })

    st.dataframe(
        missing_data,
        use_container_width=True
    )

    if missing_data["Missing Values"].sum() == 0:
        st.success("✅ Dataset không chứa dữ liệu thiếu.")
    else:
        st.warning("⚠️ Dataset có dữ liệu thiếu.")

    st.markdown("---")

    # =====================================================
    # KIỂM TRA KIỂU DỮ LIỆU
    # =====================================================

    st.subheader("🗂️ Kiểu dữ liệu")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(
        dtype_df,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # PHÂN BỐ NHÃN
    # =====================================================

    st.subheader("🎯 Phân bố nhãn")

    target_counts = (
        df["target"]
        .map({
            0: "Malignant",
            1: "Benign"
        })
        .value_counts()
    )

    st.dataframe(
        pd.DataFrame({
            "Class": target_counts.index,
            "Count": target_counts.values
        }),
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # LỰA CHỌN CỘT XEM THỐNG KÊ
    # =====================================================

    st.subheader("📊 Thống kê theo thuộc tính")

    selected_column = st.selectbox(
        "Chọn thuộc tính",
        X.columns
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Mean",
        f"{df[selected_column].mean():.2f}"
    )

    col2.metric(
        "Min",
        f"{df[selected_column].min():.2f}"
    )

    col3.metric(
        "Max",
        f"{df[selected_column].max():.2f}"
    )

    col4.metric(
        "Std",
        f"{df[selected_column].std():.2f}"
    )

    st.markdown("---")

    # =====================================================
    # TÌM KIẾM DỮ LIỆU
    # =====================================================

    st.subheader("🔎 Tìm kiếm dữ liệu")

    row_index = st.number_input(
        "Nhập chỉ số dòng",
        min_value=0,
        max_value=len(df)-1,
        value=0
    )

    st.write("Thông tin dòng được chọn:")

    st.dataframe(
        df.iloc[[row_index]],
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # DOWNLOAD DATASET
    # =====================================================

    st.subheader("⬇️ Tải dữ liệu")

    csv = df.to_csv(
        index=False
    )

    st.download_button(
        label="Download Dataset CSV",
        data=csv,
        file_name="breast_cancer_dataset.csv",
        mime="text/csv"
    )

    st.success(
        "Dataset sẵn sàng để sử dụng cho huấn luyện và đánh giá mô hình SVM."
    )