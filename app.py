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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #f4f7fa;
}

.block-container {
    padding-top: 3rem;
    padding-bottom: 2rem;
    max-width: 1380px;
}

.header-container {
    position: relative;
    padding: 70px 50px;
    border-radius: 28px;
    margin-bottom: 40px;
    background-image: linear-gradient(135deg, rgba(10, 25, 47, 0.92), rgba(23, 42, 69, 0.85)), 
                      url('https://images.unsplash.com/photo-1579154204601-01588f351167?auto=format&fit=crop&w=1200&q=80');
    background-size: cover;
    background-position: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.3);
}

.header-title {
    text-align: center;
    font-size: 46px;
    font-weight: 800;
    margin-bottom: 15px;
    background: linear-gradient(120deg, #ffffff, #a5f3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}

.header-subtitle {
    text-align: center;
    font-size: 19px;
    font-weight: 400;
    color: #94a3b8;
    max-width: 850px;
    margin: 0 auto;
    line-height: 1.6;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.04);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 20px 25px -5px rgba(59, 130, 246, 0.1), 0 10px 10px -5px rgba(59, 130, 246, 0.04);
    border-color: #06b6d4;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e2e8f0;
}

button[data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 700;
    color: #64748b;
    padding: 12px 20px;
    transition: all 0.2s ease;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #0ea5e9 !important;
    border-bottom: 3px solid #0ea5e9 !important;
}

.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 52px;
    font-weight: 700;
    font-size: 16px;
    background: linear-gradient(135deg, #0ea5e9, #2563eb);
    color: white;
    border: none;
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #38bdf8, #3b82f6);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4);
    transform: translateY(-1px);
}

.stButton > button:active {
    transform: scale(0.98);
}

.input-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 30px;
    margin-top: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

div.element-container stAlert {
    border-radius: 16px;
    font-weight: 500;
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
        st.markdown("<h2 style='font-weight:800; color:#1e293b; margin-bottom:20px;'>⚙️ Control Center</h2>", unsafe_allow_html=True)
        st.success("Kết nối dữ liệu thành công")
        st.markdown("---")
        st.metric("Tổng số mẫu (Samples)", f"{df.shape[0]:,}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Đặc trưng đầu vào (Features)", X.shape[1])

    st.markdown(f"""
    <div class="header-container">
        <div class="header-title">🧠 SVM Breast Cancer Analysis Studio</div>
        <div class="header-subtitle">Nền tảng ứng dụng thuật toán học máy Vector Hỗ Trợ (Support Vector Machine) vào chẩn đoán và phân tích dữ liệu lâm sàng ung thư vú công nghệ cao</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📥 Huấn luyện & Dự đoán", "📊 Trực quan hóa sinh động", "📋 Khảo sát Dataset"])

    with tab1:
        st.markdown("<h3 style='font-weight:700; color:#0f172a; margin-bottom:15px;'>🚀 Mô hình phân lớp SVM Pipeline</h3>", unsafe_allow_html=True)
        
        if st.button("Khai hỏa quy trình huấn luyện hệ thống SVM", use_container_width=True):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = SVC(probability=True, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            train_pred = model.predict(X_train_scaled)
            test_pred = model.predict(X_test_scaled)
            
            st.success("🎉 Hệ thống SVM đã hoàn tất phân tích và tối ưu hóa trọng số không gian!")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Độ chính xác (Accuracy Train / Test)", f"{accuracy_score(y_train, train_pred)*100:.2f}% / {accuracy_score(y_test, test_pred)*100:.2f}%")
            m2.metric("Hệ số tương quan (R² Train / Test)", f"{r2_score(y_train, train_pred):.4f} / {r2_score(y_test, test_pred):.4f}")
            m3.metric("Sai số tuyệt đối (MAE Train / Test)", f"{mean_absolute_error(y_train, train_pred):.4f} / {mean_absolute_error(y_test, test_pred):.4f}")

            st.session_state['trained_svm'] = model
            st.session_state['svm_scaler'] = scaler

        if 'trained_svm' in st.session_state:
            st.markdown("<br><hr>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-weight:700; color:#0f172a; margin-bottom:10px;'>🎯 Trình kiểm tra mẫu bệnh án lâm sàng</h3>", unsafe_allow_html=True)
            
            st.markdown('<div class="input-card">', unsafe_allow_html=True)
            feature_inputs = {}
            cols = st.columns(min(4, len(X.columns)))
            for idx, col_name in enumerate(X.columns):
                with cols[idx % len(cols)]:
                    feature_inputs[col_name] = st.number_input(
                        f"{col_name}", 
                        value=float(X[col_name].mean())
                    )
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Thể hiện kết quả dự đoán lâm sàng từ SVM", use_container_width=True):
                sample_data = np.array([list(feature_inputs.values())])
                sample_scaled = st.session_state['svm_scaler'].transform(sample_data)
                
                prediction = st.session_state['trained_svm'].predict(sample_scaled)[0]
                probabilities = st.session_state['trained_svm'].predict_proba(sample_scaled)[0]
                max_prob = np.max(probabilities)
                
                label_mapping = {0: "Mác ác tính (M)", 1: "Lành tính (B)"} if df[target_col].iloc[0] == 'M' else {0: "Lành tính (B)", 1: "Mác ác tính (M)"}
                pred_label = label_mapping.get(prediction, f"Nhãn {prediction}")
                
                st.info(f"🔮 **Kết quả nhận diện phân lớp tự động:** {pred_label} — Độ tin cậy thuật toán đạt: **{max_prob*100:.2f}%**")

    with tab2:
        st.markdown("<h3 style='font-weight:700; color:#0f172a; margin-bottom:20px;'>📊 Hệ thống đồ thị khai phá dữ liệu đa phong cách</h3>", unsafe_allow_html=True)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        df_vis = df.copy()
        df_vis[target_col] = y
        
        st.subheader("1️⃣ Phong cách Tần suất (Histogram) - Phân phối thuộc tính")
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
        fig1.update_layout(margin=dict(l=20, r=20, t=20, b=20), borderRadius=15)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("---")

        st.subheader("2️⃣ Phong cách Đối sánh Tập hợp (Boxplot) - Phát hiện Outliers")
        selected_box_col = st.selectbox("Chọn thuộc tính cần đối sánh biên độ dữ liệu:", numeric_cols, index=min(1, len(numeric_cols)-1))
        fig2 = px.box(
            df_vis, 
            y=selected_box_col, 
            x=target_col, 
            color=target_col,
            points="all",
            notched=True,
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig2.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("---")

        st.subheader("3️⃣ Phong cách Ma trận Mật độ (Heatmap) - Trọng số tương quan")
        if len(numeric_cols) > 1:
            corr_matrix = df_vis[numeric_cols + [target_col]].corr()
            fig3, ax = plt.subplots(figsize=(12, 7))
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap="YlGnBu", 
                fmt=".2f", 
                linewidths=0.8, 
                linecolor="#ffffff",
                cbar=True,
                ax=ax
            )
            ax.set_title("Ma trận biểu diễn tương quan giữa các chiều biến chứng", fontsize=13, pad=12, weight='bold', color='#0f172a')
            st.pyplot(fig3)
        else:
            st.warning("Không đủ thuộc tính số để phân tích mối tương quan.")
        st.markdown("---")

        st.subheader("4️⃣ Phong cách Phân tách Không gian (Scatter Plot) - Khảo sát biên giới SVM")
        if len(numeric_cols) >= 2:
            scat_col1 = st.selectbox("Chọn đặc trưng trục X:", numeric_cols, index=0)
            scat_col2 = st.selectbox("Chọn đặc trưng trục Y:", numeric_cols, index=min(1, len(numeric_cols)-1))
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
            st.warning("Cần tối thiểu 2 trường dữ liệu số để phân hóa không gian biểu đồ.")
        st.markdown("---")

        st.subheader("5️⃣ Phong cách Tỉ lệ Phân chia (Donut Chart) - Cân bằng nhãn mục tiêu")
        target_counts = df_vis[target_col].value_counts().reset_index()
        target_counts.columns = ['Nhãn lớp', 'Số lượng mẫu']
        fig5 = px.pie(
            target_counts, 
            values='Số lượng mẫu', 
            names='Nhãn lớp', 
            hole=0.45,
            template="plotly_white",
            color_discrete_sequence=["#0ea5e9", "#f43f5e"]
        )
        fig5.update_traces(textinfo='percent+label', pull=[0.02] * len(target_counts))
        fig5.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig5, use_container_width=True)

except FileNotFoundError:
    st.error("❌ Không tìm thấy tập tin dữ liệu 'breast_cancer.csv'.")
