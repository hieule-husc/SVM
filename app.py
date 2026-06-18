import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    target_col = df.columns[-1]
    
    if y.dtype == 'object':
        y = pd.factorize(y)[0]
        df[target_col] = y

    with st.sidebar:
        st.title("⚙️ SVM Status")
        st.success("Đã kết nối breast_cancer.csv")
        st.metric("Tổng số mẫu", df.shape[0])
        st.metric("Số đặc trưng đầu vào", X.shape[1])

    st.markdown(f"""
    <div class="header-container">
        <div class="header-title">⚡ SVM Comprehensive Evaluation Studio</div>
        <div class="header-subtitle">Hệ thống tối ưu hóa và đánh giá hiệu năng thuật toán Support Vector Machine (SVM)</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📥 Huấn luyện & Dự đoán", "📊 Trực quan hóa", "📋 Xem Dataset"])

    with tab1:
        st.subheader("🚀 Khởi chạy mô hình học máy SVM")
        
        if st.button("Chạy huấn luyện hệ thống SVM", use_container_width=True):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            st.success("Hệ thống SVM đã phân tích xong dữ liệu từ breast_cancer.csv!")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Accuracy (Train / Test)", f"{accuracy_score(y_train, train_pred)*100:.2f}% / {accuracy_score(y_test, test_pred)*100:.2f}%")
            m2.metric("R² Score (Train / Test)", f"{r2_score(y_train, train_pred):.4f} / {r2_score(y_test, test_pred):.4f}")
            m3.metric("MAE (Train / Test)", f"{mean_absolute_error(y_train, train_pred):.4f} / {mean_absolute_error(y_test, test_pred):.4f}")

            st.session_state['trained_svm'] = model
            st.session_state['svm_scaler'] = scaler

        if 'trained_svm' in st.session_state:
            st.markdown("---")
            st.subheader("🎯 Nhập thông số mẫu mới để kiểm tra dự đoán")
            
            feature_inputs = {}
            cols = st.columns(min(4, len(X.columns)))
            for idx, col_name in enumerate(X.columns):
                with cols[idx % len(cols)]:
                    feature_inputs[col_name] = st.number_input(
                        f"{col_name}", 
                        value=float(X[col_name].mean())
                    )
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Thể hiện kết quả dự đoán của mẫu", use_container_width=True):
                sample_data = np.array([list(feature_inputs.values())])
                sample_scaled = st.session_state['svm_scaler'].transform(sample_data)
                
                prediction = st.session_state['trained_svm'].predict(sample_scaled)[0]
                probabilities = st.session_state['trained_svm'].predict_proba(sample_scaled)[0]
                max_prob = np.max(probabilities)
                
                st.info(f"Kết quả dự đoán phân lớp từ SVM: **Nhãn {prediction}** (Độ tin cậy: {max_prob*100:.2f}%)")

    with tab2:
        st.header("📊 Hệ thống 5 Biểu đồ Trực quan hóa Đa phong cách")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        st.subheader("1️⃣ Phong cách Tần suất (Histogram) - Phân phối thuộc tính đầu vào")
        selected_hist_col = st.selectbox("Chọn thuộc tính cần xem phân phối:", numeric_cols, index=0)
        fig1 = px.histogram(
            df, 
            x=selected_hist_col, 
            color=target_col,
            marginal="rug",
            barmode="overlay",
            title=f"Biểu đồ phân phối tần suất của {selected_hist_col} phân tách theo nhãn target",
            template="plotly_white"
        )
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("---")

        st.subheader("2️⃣ Phong cách Đối sánh Tập hợp (Boxplot) - Phát hiện Outliers & Biên độ")
        selected_box_col = st.selectbox("Chọn thuộc tính cần đối sánh biên độ dữ liệu:", numeric_cols, index=min(1, len(numeric_cols)-1))
        fig2 = px.box(
            df, 
            y=selected_box_col, 
            x=target_col, 
            color=target_col,
            points="all",
            notched=True,
            title=f"Biểu đồ hộp phân tích độ phân tán và ngoại lai của {selected_box_col} qua từng nhóm nhãn",
            template="seaborn"
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")

        st.subheader("3️⃣ Phong cách Ma trận Mật độ (Heatmap) - Đo lường độ tương quan tuyến tính")
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols + [target_col]].corr()
            fig3, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap="coolwarm", 
                fmt=".2f", 
                linewidths=1, 
                linecolor="white",
                cbar=True,
                ax=ax
            )
            plt.title("Ma trận biểu diễn trọng số tương quan (Correlation Matrix) giữa các đặc trưng", fontsize=14, pad=15)
            st.pyplot(fig3)
        else:
            st.warning("Không đủ số lượng cột dữ liệu số để thiết lập biểu đồ ma trận tương quan.")
        st.markdown("---")

        st.subheader("4️⃣ Phong cách Phân tách Không gian (Scatter Plot) - Khảo sát ranh giới phân lớp SVM")
        if len(numeric_cols) >= 2:
            scat_col1 = st.selectbox("Chọn thuộc tính trục X:", numeric_cols, index=0)
            scat_col2 = st.selectbox("Chọn thuộc tính trục Y:", numeric_cols, index=min(1, len(numeric_cols)-1))
            fig4 = px.scatter(
                df, 
                x=scat_col1, 
                y=scat_col2, 
                color=target_col,
                size=df[scat_col1].abs() if df[scat_col1].min() >= 0 else None,
                hover_data=numeric_cols[:3],
                title=f"Không gian phân tách dữ liệu thực tế giữa {scat_col1} và {scat_col2}",
                template="plotly_dark"
            )
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Cần tối thiểu 2 trường dữ liệu số để thiết lập biểu đồ phân tách không gian.")
        st.markdown("---")

        st.subheader("5️⃣ Phong cách Tỉ lệ Phân chia (Pie Chart) - Cân bằng nhãn lớp Target")
        target_counts = df[target_col].value_counts().reset_index()
        target_counts.columns = ['Nhãn lớp', 'Số lượng mẫu']
        fig5 = px.pie(
            target_counts, 
            values='Số lượng mẫu', 
            names='Nhãn lớp', 
            hole=0.4,
            color='Nhãn lớp',
            title="Tỉ lệ phần trăm cấu trúc số lượng các lớp trong bài toán phân loại nhãn mục tiêu",
            template="ggplot2"
        )
        fig5.update_traces(textinfo='percent+label', pull=[0.05] * len(target_counts))
        st.plotly_chart(fig5, use_container_width=True)

    with tab3:
        st.header("📋 Dataset Explorer")
        st.subheader("📄 10 dòng đầu tiên của file breast_cancer.csv")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        st.subheader("📈 Thống kê mô tả tổng quan")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("---")
        st.subheader("🔍 Kiểm tra các giá trị trống (Missing Values)")
        missing_data = pd.DataFrame({"Trường dữ liệu": df.columns, "Số lượng ô trống": df.isnull().sum().values})
        st.dataframe(missing_data, use_container_width=True)

except FileNotFoundError:
    st.error("❌ Không tìm thấy file 'breast_cancer.csv'. Vui lòng đặt file dữ liệu này nằm cùng cấp với file chạy ứng dụng Streamlit.")
