import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Dự đoán thể trạng Phân Nhóm", layout="centered")

# 1. Từ điển để map kết quả dự đoán (0, 1, 2) ra chữ hiển thị
INDEX_MAP = {
    0: "Gầy",
    1: "Bình thường",
    2: "Mập"
}

# 2. Hàm để tải mô hình đã lưu
@st.cache_resource  # Dùng cache của Streamlit để không bị load lại mô hình mỗi khi người dùng thao tác
def load_model():
    with open('gnb_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Tải mô hình
model = load_model()

# 3. Xây dựng giao diện Web
st.title("Hệ thống Phân loại Thể trạng cơ thể (GNB)")
st.write("Thuật toán dự đoán dựa trên 3 nhóm: Gầy, Bình thường và Mập.")

# Tạo form nhập liệu
with st.form("prediction_form"):
    st.subheader("Nhập thông tin của bạn:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_input = st.selectbox("Giới tính", ("Nam", "Nữ"))
    with col2:
        height_input = st.number_input("Chiều cao (cm)", min_value=50.0, max_value=250.0, value=170.0, step=1.0)
    with col3:
        weight_input = st.number_input("Cân nặng (kg)", min_value=10.0, max_value=200.0, value=65.0, step=1.0)
    submit_button = st.form_submit_button(label="Dự đoán Thể trạng")

# 4. Xử lý logic khi người dùng bấm nút
if submit_button:
    # Mã hóa dữ liệu đầu vào sao cho khớp với chuẩn lúc huấn luyện (Nam=0, Nữ=1)
    gender_encoded = 0 if gender_input == "Nam" else 1
    
    # Tính toán BMI
    bmi = weight_input / ((height_input / 100) ** 2)
    
    # Tạo DataFrame chứa thông tin người dùng truyền vào mô hình
    input_data = pd.DataFrame(
        [[gender_encoded, height_input, weight_input, bmi]], 
        columns=['Gender', 'Height', 'Weight', 'BMI']
    )
    
    # Dự đoán bằng mô hình đã load
    prediction_value = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    # Lấy xác suất tự tin cao nhất tương ứng với class dự đoán
    # get the index of prediction in model.classes_
    class_index = list(model.classes_).index(prediction_value)
    confidence = probabilities[class_index] * 100
    
    # Hiển thị kết quả
    st.success(f"### Kết quả phân loại: {INDEX_MAP[prediction_value]}")
    st.info(f"Độ tự tin của mô hình Gaussian Naive Bayes: **{confidence:.2f}%**")
    
    # Hiển thị biểu đồ thanh cho xác suất của tất cả các nhóm
    st.write("Chi tiết xác suất từng nhóm:")
    prob_df = pd.DataFrame({
        "Nhóm": ["Gầy", "Bình thường", "Mập"],
        "Xác suất (%)": probabilities * 100
    })
    st.bar_chart(prob_df.set_index("Nhóm"))