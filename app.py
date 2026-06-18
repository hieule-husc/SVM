import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error

st.set_page_config(
    page_title="SVM Evaluation System",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
.block-container{padding-top:1rem;padding-bottom:1rem;max-width:1400px;}
.header-container{padding:30px;border-radius:20px;margin-bottom:20px;background:linear-gradient(135deg, rgba(37,99,235,0.12), rgba(59,130,246,0.03));border:1px solid rgba(128,128,128,0.15);}
.header-title{text-align:center;font-size:40px;font-weight:800;margin-bottom:10px;color:inherit;}
.header-subtitle{text-align:center;font-size:20px;font-weight:500;color:inherit;opacity:0.9;}
div[data-testid="stMetric"]{background:rgba(128,128,128,0.05);border:1px solid rgba(128,128,128,0.15);border-radius:15px;padding:15px;box-shadow:0 2px 10px rgba(0,0,0,0.05);}
div[data-testid="stMetric"]:hover{transform:translateY(-2px);transition:0.25s ease;}
section[data-testid="stSidebar"]{border-right:1px solid rgba(128,128,128,0.15);}
button[data-baseweb="tab"]{font-size:16px;font-weight:600;}
button[data-baseweb="tab"][aria-selected="true"]{color:#2563EB;border-bottom:3px solid #2563EB;}
.stButton > button{width:100%;border-radius:10px;height:45px;font-weight:600;}
.stDownloadButton > button{width:100%;border-radius:10px;font-weight:600;}
[data-testid="stDataFrame"]{border-radius:12px;}
[data-testid="stSuccess"]{border-radius:12px;}
[data-testid="stInfo"]{border-radius:12px;}
[data-testid="stError"]{border-radius:12px;}
::-webkit-scrollbar{width:10px;}
::-webkit-scrollbar-thumb{background:#2563EB;border-radius:20px;}
</style>
""", unsafe_allow_html=True)

try:
    df = pd.read_csv("breast_cancer.csv")
    
    with st.sidebar:
        st.title("⚙️ SVM Data Input")
        st.success("Đã tìm thấy file breast_cancer.csv")
        columns_list = list(df.columns)
        target_col = st.selectbox("Chọn cột Target (Nhãn)", columns_list, index=len(columns_list)-1)
        st.markdown("---")
        st.info("""
        **Hệ thống phân tích SVM**
        Dữ liệu đã nạp sẵn từ tệp cấu hình của bạn. Chỉnh sửa tham số huấn luyện để kiểm tra hiệu năng.
        """)

    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    is_classification = len(np.unique(y)) <= 10 or y.dtype == 'object'
    if is_classification and y.dtype == 'object':
        y = pd.factorize(y)[0]

    st.markdown(f"""
    <div class="header-container">
        <div class="header-title">⚡ SVM Comprehensive Evaluation Studio</div>
        <div class="header-subtitle">Hệ thống tối ưu hóa và đánh giá hiệu năng thuật toán Support Vector Machine (SVM)</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📥 Huấn luyện & Dự đoán", "📊 Trực quan hóa", "📋 Xem Dataset"])

    with tab1:
        st.subheader("⚙️ Cấu hình Hyperparameters cho SVM")
        
        col1, col2 = st.columns(2)
        with col1:
            C = st.slider("Regularization Parameter (C)", 0.1, 100.0, 1.0)
            gamma = st.selectbox("Gamma", ["scale", "auto"])
        with col2:
            test_size = st.slider("Tỷ lệ Test Size", 0.1, 0.5, 0.2)
            kernel_type = st.selectbox("Cơ chế Kernel SVM", ["rbf", "linear", "poly", "sigmoid"])

        degree = 3
        if kernel_type == "poly":
            degree = st.slider("Polynomial Degree", 2, 10, 3)

        if st.button("🚀 Thực thi Huấn luyện SVM", use_container_width=True):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            if is_classification:
                model = SVC(kernel=kernel_type, C=C, gamma=gamma, degree=degree, probability=True)
                model.fit(X_train_scaled, y_train)
                
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                acc_train = accuracy_score(y_train, train_pred)
                acc_test = accuracy_score(y_test, test_pred)
                
                r2_train = r2_score(y_train, train_pred)
                r2_test = r2_score(y_test, test_pred)
                
                mae_train = mean_absolute_error(y_train, train_pred)
                mae_test = mean_absolute_error(y_test, test_pred)
                
                st.success("Huấn luyện Mô hình SVM Phân loại thành công!")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Accuracy (Train / Test)", f"{acc_train*100:.2f}% / {acc_test*100:.2f}%")
                m2.metric("R² Score (Train / Test)", f"{r2_train:.4f} / {r2_test:.4f}")
                m3.metric("MAE (Train / Test)", f"{mae_train:.4f} / {mae_test:.4f}")
            else:
                model = SVR(kernel=kernel_type, C=C, gamma=gamma, degree=degree)
                model.fit(X_train_scaled, y_train)
                
                train_pred = model.predict(X_train_scaled)
                test_pred = model.predict(X_test_scaled)
                
                r2_train = r2_score(y_train, train_pred)
                r2_test = r2_score(y_test, test_pred)
                
                mae_train = mean_absolute_error(y_train, train_pred)
                mae_test = mean_absolute_error(y_test, test_pred)
                
                st.success("Huấn luyện Mô hình SVM Hồi quy thành công!")
                
                m1, m2 = st.columns(2)
                m1.metric("R² Score (Train / Test)", f"{r2_train:.4f} / {r2_test:.4f}")
                m2.metric("MAE (Train / Test)", f"{mae_train:.4f} / {mae_test:.4f}")

            st.session_state['trained_svm'] = model
            st.session_state['svm_scaler'] = scaler
            st.session_state['features_columns'] = list(X.columns)
            st.session_state['is_classification'] = is_classification

        if 'trained_svm' in st.session_state:
            st.markdown("---")
            st.subheader("🎯 Dự đoán dữ liệu mẫu mới")
            
            feature_inputs = {}
            cols = st.columns(min(4, len(X.columns)))
            for idx, col_name in enumerate(X.columns):
                with cols[idx % len(cols)]:
                    feature_inputs[col_name] = st.number_input(f"{col_name}", value=float(X[col_name].mean()))
            
            if st.button("Predict Custom Sample"):
                sample_data = np.array([list(feature_inputs.values())])
                sample_scaled = st.session_state['svm_scaler'].transform(sample_data)
                
                prediction = st.session_state['trained_svm'].predict(sample_scaled)[0]
                
                if st.session_state['is_classification']:
                    st.info(f"Kết quả dự đoán phân lớp nhãn: **{prediction}**")
                else:
                    st.info(f"Kết quả dự đoán giá trị hồi quy: **{prediction:.4f}**")

    with tab2:
        st.header("📊 Trực quan hóa dữ liệu")
        
        st.subheader("1️⃣ Biểu đồ phân bổ thuộc tính Target")
        if is_classification:
            target_counts = y.value_counts()
            fig1 = px.bar(x=target_counts.index, y=target_counts.values, color=target_counts.index, text=target_counts.values)
            fig1.update_layout(xaxis_title="Class", yaxis_title="Số lượng mẫu")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            fig1 = px.histogram(df, x=target_col, nbins=30, marginal="box")
            st.plotly_chart(fig1, use_container_width=True)

        st.markdown("---")
        st.subheader("2️⃣ Ma trận tương quan Features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            fig2, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="RdBu_r", fmt=".2f", linewidths=0.5, ax=ax)
            st.pyplot(fig2)
        else:
            st.warning("Không đủ thuộc tính số để vẽ ma trận tương quan.")

    with tab3:
        st.header("📋 Dataset Explorer")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Số lượng dòng (Samples)", df.shape[0])
        c2.metric("Số lượng cột", df.shape[1])
        c3.metric("Số lượng đặc trưng (Features)", X.shape[1])
        
        st.markdown("---")
        st.subheader("📄 10 dòng đầu tiên của dữ liệu")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 Thống kê mô tả tổng quan")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Kiểm tra dữ liệu thiếu")
        missing_data = pd.DataFrame({"Column": df.columns, "Missing Values": df.isnull().sum().values})
        st.dataframe(missing_data, use_container_width=True)

except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'breast_cancer.csv'. Vui lòng đảm bảo file CSV nằm cùng thư mục chạy ứng dụng Streamlit của bạn.")
