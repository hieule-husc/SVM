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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif;
    background-color: #f8fafc;
}
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2rem;
    max-width: 1350px;
}
.header-container {
    position: relative;
    padding: 60px 40px;
    border-radius: 24px;
    margin-bottom: 35px;
    background-image: linear-gradient(135deg, rgba(15, 23, 42, 0.85), rgba(30, 41, 59, 0.75)), 
                      url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1200&q=80');
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
}
.header-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 12px;
    color: #ffffff;
    letter-spacing: -0.025em;
}
.header-subtitle {
    text-align: center;
    font-size: 19px;
    font-weight: 400;
    color: #cbd5e1;
    max-width: 800px;
    margin: 0 auto;
    line-height: 1.5;
}
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.15);
    border-color: #3b82f6;
}
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}
button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
    color: #64748b;
    padding-bottom: 8px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 3px solid #3b82f6 !important;
}
.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 48px;
    font-weight: 600;
    background-color: #3b82f6;
    color: white;
    border: none;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background-color: #2563eb;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
}
.stButton > button:active {
    transform: scale(0.98);
}
[data-testid="stDataFrame"], [data-testid="stSubheader"] {
    margin-top: 10px;
}
div.element-container stAlert {
    border-radius: 14px;
}
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #f1f5f9;
}
::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 20px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

try:
    df = pd.read_csv("breast_cancer.csv")
    
    target_col = "diagnosis"
    
    if "id" in df.columns:
        X = df.drop(columns=["id", target_col])
    else:
        X = df.drop(columns=[target_col])
        
    y = df[target_col]
    
    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    with st.sidebar:
        st.title("⚙️ SVM Control Panel")
        st.success("Đã kết nối breast_cancer.csv")
        st.markdown("---")
        st.metric("Tổng mẫu lưu trữ (Samples)", f"{df.shape[0]:,}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Số đặc trưng lọc (Features)", X.shape[1])

    st.markdown(f"""
    <div class="header-container">
        <div class="header-title">🧠 SVM Cancer Diagnosis Analyzer</div>
        <div class="header-subtitle">Hệ thống phân tích chuyên sâu và hỗ trợ dự đoán ung thư vú áp dụng kiến trúc thuật toán máy học phân lớp tối ưu Support Vector Machine (SVM)</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📥 Huấn luyện & Dự đoán", "📊 Trực quan hóa dữ liệu", "📋 Thống kê Dataset"])

    with tab1:
        st.subheader("🚀 Tiến trình phân tích & Huấn luyện")
        
        if st.button("Khai hỏa quy trình huấn luyện SVM Pipeline", use_container_width=True):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            st.success("Hệ thống mạng học máy SVM đã tối ưu và hội tụ thành công!")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Mức chính xác (Accuracy Train / Test)", f"{accuracy_score(y_train, train_pred)*100:.2f}% / {accuracy_score(y_test, test_pred)*100:.2f}%")
            m2.metric("Hệ số xác định (R² Train / Test)", f"{r2_score(y_train, train_pred):.4f} / {r2_score(y_test, test_pred):.4f}")
            m3.metric("Sai số tuyệt đối (MAE Train / Test)", f"{mean_absolute_error(y_train, train_pred):.4f} / {mean_absolute_error(y_test, test_pred):.4f}")

            st.session_state['trained_svm'] = model
            st.session_state['svm_scaler'] = scaler

        if 'trained_svm' in st.session_state:
            st.markdown("<br><hr>", unsafe_allow_html=True)
            st.subheader("🎯 Cửa sổ dự nghiệm mẫu bệnh án mới")
            
            feature_inputs = {}
            cols = st.columns(min(4, len(X.columns)))
            for idx, col_name in enumerate(X.columns):
                with cols[idx % len(cols)]:
                    feature_inputs[col_name] = st.number_input(
                        f"{col_name}", 
                        value=float(X[col_name].mean())
                    )
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Tính toán & Xuất kết quả dự đoán lâm sàng", use_container_width=True):
                sample_data = np.array([list(feature_inputs.values())])
                sample_scaled = st.session_state['svm_scaler'].transform(sample_data)
                
                prediction = st.session_state['trained_svm'].predict(sample_scaled)[0]
                probabilities = st.session_state['trained_svm'].predict_proba(sample_scaled)[0]
                max_prob = np.max(probabilities)
                
                label_mapping = {0: "Mác ác tính (M)", 1: "Lành tính (B)"} if df[target_col].iloc[0] == 'M' else {0: "Lành tính (B)", 1: "Mác ác tính (M)"}
                pred_label = label_mapping.get(prediction, f"Nhãn {prediction}")
                
                st.info(f"Kết quả nhận diện tự động từ bộ phân lớp SVM: **{pred_label}** — Xác suất tin cậy cấu trúc: **{max_prob*100:.2f}%**")

    with tab2:
        st.subheader("📊 Hệ thống 5 Biểu đồ Khai phá Đa góc nhìn")
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        df_vis = df.copy()
        df_vis[target_col] = y
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("1️⃣ Phong cách Tần suất (Histogram) - Phân phối thuộc tính đầu vào")
        selected_hist_col = st.selectbox("Chọn thuộc tính cần xem phân phối:", numeric_cols, index=0)
        fig1 = px.histogram(
            df_vis, 
            x=selected_hist_col, 
            color=target_col,
            marginal="rug",
            barmode="overlay",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig1.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("---")

        st.subheader("2️⃣ Phong cách Đối sánh Tập hợp (Boxplot) - Phát hiện Outliers & Biên độ")
        selected_box_col = st.selectbox("Chọn thuộc tính cần đối sánh biên độ dữ liệu:", numeric_cols, index=min(1, len(numeric_cols)-1))
        fig2 = px.box(
            df_vis, 
            y=selected_box_col, 
            x=target_col, 
            color=target_col,
            points="all",
            notched=True,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Modern
        )
        fig2.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")

        st.subheader("3️⃣ Phong cách Ma trận Mật độ (Heatmap) - Đo lường độ tương quan tuyến tính")
        if len(numeric_cols) > 1:
            corr_matrix = df_vis[numeric_cols + [target_col]].corr()
            fig3, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap="YlGnBu", 
                fmt=".2f", 
                linewidths=0.8, 
                linecolor="#f8fafc",
                cbar=True,
                ax=ax
            )
            ax.set_title("Ma trận biểu diễn trọng số tương quan giữa các biến hệ thống", fontsize=13, pad=12, weight='bold')
            st.pyplot(fig3)
        else:
            st.warning("Không đủ số lượng cột dữ liệu số để thiết lập biểu đồ ma trận tương quan.")
        st.markdown("---")

        st.subheader("4️⃣ Phong cách Phân tách Không gian (Scatter Plot) - Khảo sát ranh giới phân lớp SVM")
        if len(numeric_cols) >= 2:
            scat_col1 = st.selectbox("Chọn thuộc tính trục X:", numeric_cols, index=0)
            scat_col2 = st.selectbox("Chọn thuộc tính trục Y:", numeric_cols, index=min(1, len(numeric_cols)-1))
            fig4 = px.scatter(
                df_vis, 
                x=scat_col1, 
                y=scat_col2, 
                color=target_col,
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            fig4.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.warning("Cần tối thiểu 2 trường dữ liệu số để thiết lập biểu đồ phân tách không gian.")
        st.markdown("---")

        st.subheader("5️⃣ Phong cách Tỉ lệ Phân chia (Pie Chart) - Cân bằng nhãn lớp Target")
        target_counts = df_vis[target_col].value_counts().reset_index()
        target_counts.columns = ['Nhãn lớp', 'Số lượng mẫu']
        fig5 = px.pie(
            target_counts, 
            values='Số lượng mẫu', 
            names='Nhãn lớp', 
            hole=0.45,
            template="plotly_white",
            color_discrete_sequence=["#3b82f6", "#ef4444"]
        )
        fig5.update_traces(textinfo='percent+label', pull=[0.02] * len(target_counts))
        fig5.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig5, use_container_width=True)

    with tab3:
        st.subheader("📋 Phân tích dữ liệu gốc")
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("📊 **10 bản ghi dữ liệu lâm sàng đầu tiên:**")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.write("📈 **Bảng thống kê thuộc tính phân phối mô tả:**")
        st.dataframe(df.describe(), use_container_width=True)
        
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.write("🔍 **Báo cáo mật độ dữ liệu khuyết thiếu (Missing Values):**")
        missing_data = pd.DataFrame({"Trường thuộc tính số": df.columns, "Số ô trống": df.isnull().sum().values})
        st.dataframe(missing_data, use_container_width=True)

except FileNotFoundError:
    st.error("❌ Hệ thống không tìm thấy tập tin 'breast_cancer.csv'. Vui lòng kiểm tra lại đường dẫn lưu trữ thư mục gốc.")
