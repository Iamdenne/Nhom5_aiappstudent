import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Student Survival App", page_icon="🎓", layout="wide")

# Đường dẫn thư mục hiện tại (trên Cloud)
MODEL_DIR = "."

@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    metadata = joblib.load(os.path.join(MODEL_DIR, "model_metadata.pkl"))
    return scaler, model, metadata

scaler, model, metadata = load_artifacts()

def predict_risk(credits, gpa, study_h, sleep_h, absence, part_time_h, stress, assign, mid, scheme_str):
    w_assign = 0.2
    w_mid = 0.3 if scheme_str == "0.2 - 0.3 - 0.5" else 0.2
    w_final = 0.5 if scheme_str == "0.2 - 0.3 - 0.5" else 0.6
    scheme_encoded = 0 if scheme_str == "0.2 - 0.3 - 0.5" else 1

    current_weighted = assign * w_assign + mid * w_mid
    mid_assign_diff = mid - assign
    min_final_pass = np.clip((4.0 - current_weighted) / w_final, 0.0, 15.0)

    input_df = pd.DataFrame([{
        'Current_Credits': credits,
        'Cumulative_GPA': gpa,
        'Study_Hours_Per_Week': study_h,
        'Sleep_Hours_Per_Day': sleep_h,
        'Absence_Rate': absence,
        'Part_Time_Hours_Per_Week': part_time_h,
        'Stress_Level': stress,
        'Assignment_Score': assign,
        'Midterm_Score': mid,
        'Scheme_Encoded': scheme_encoded,
        'Current_Weighted_Score': current_weighted,
        'Mid_Assign_Diff': mid_assign_diff,
        'Min_Final_Needed_For_Pass': min_final_pass
    }])

    input_scaled = scaler.transform(input_df)
    pred_label = model.predict(input_scaled)[0]
    pred_probs = model.predict_proba(input_scaled)[0]

    t_pass = max(0.0, (4.0 - current_weighted) / w_final)
    t_good = max(0.0, (7.0 - current_weighted) / w_final)
    t_exc = max(0.0, (8.5 - current_weighted) / w_final)

    return pred_label, pred_probs, current_weighted, t_pass, t_good, t_exc

st.title("🎓 AI Student Survival App - Hệ Thống Cảnh Báo Sớm Rủi Ro Học Tập")
st.caption(f"Mô hình lõi: **{metadata['best_model_name']}** | Độ chính xác (Test Accuracy): **{metadata['test_accuracy']*100:.2f}%**")

col_in, col_out = st.columns([1.2, 1.3], gap="large")

with col_in:
    st.subheader("1. Thông tin học tập & Sinh hoạt")
    c1, c2 = st.columns(2)
    credits = c1.number_input("Số tín chỉ đang học:", min_value=1, max_value=30, value=18)
    gpa = c2.number_input("Điểm GPA tích lũy (hệ 4):", min_value=0.0, max_value=4.0, value=2.8, step=0.1)
    study_h = c1.number_input("Số giờ tự học / tuần:", min_value=0.0, max_value=60.0, value=10.0, step=0.5)
    sleep_h = c2.number_input("Số giờ ngủ / ngày:", min_value=2.0, max_value=14.0, value=6.5, step=0.5)
    absence = c1.slider("Tỷ lệ vắng mặt trên lớp (%):", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
    part_time_h = c2.number_input("Thời gian làm thêm (giờ/tuần):", min_value=0.0, max_value=50.0, value=8.0, step=1.0)
    stress = st.slider("Mức độ căng thẳng tự đánh giá (1 - 10):", min_value=1, max_value=10, value=5)

    st.subheader("2. Điểm thành phần môn học")
    s1, s2, s3 = st.columns(3)
    scheme = s1.selectbox("Hệ số (BT-GK-CK):", ["0.2 - 0.3 - 0.5", "0.2 - 0.2 - 0.6"])
    assign = s2.number_input("Điểm Bài tập:", min_value=0.0, max_value=10.0, value=8.5, step=0.1)
    mid = s3.number_input("Điểm Giữa kỳ:", min_value=0.0, max_value=10.0, value=7.0, step=0.1)

    btn = st.button("🔍 Dự đoán trạng thái rủi ro", type="primary", use_container_width=True)

with col_out:
    st.subheader("3. Kết quả phân tích & Khuyến nghị")
    if btn:
        label, probs, curr_score, t_pass, t_good, t_exc = predict_risk(
            credits, gpa, study_h, sleep_h, absence, part_time_h, stress, assign, mid, scheme
        )
        if label == 0:
            st.success(f"### 🟢 AN TOÀN (SAFE) — Xác suất: {probs[0]*100:.1f}%")
        elif label == 1:
            st.warning(f"### 🟡 CẢNH BÁO (WARNING) — Xác suất: {probs[1]*100:.1f}%")
        else:
            st.error(f"### 🔴 NGUY HIỂM - NGUY CƠ RỚT MÔN (AT RISK) — Xác suất: {probs[2]*100:.1f}%")

        st.metric("Điểm tích lũy quá trình hiện tại", f"{curr_score:.2f} / 10.0")
        m1, m2, m3 = st.columns(3)
        m1.metric("Qua môn (≥ 4.0)", f"{t_pass:.2f} đ" if t_pass <= 10 else "Không thể (>10)")
        m2.metric("Đạt Khá (≥ 7.0)", f"{t_good:.2f} đ" if t_good <= 10 else "Không khả thi")
        m3.metric("Đạt Giỏi (≥ 8.5)", f"{t_exc:.2f} đ" if t_exc <= 10 else "Không khả thi")

        prob_df = pd.DataFrame({
            "Trạng thái": ["An toàn (Safe)", "Cảnh báo (Warning)", "Nguy cơ rớt (At Risk)"],
            "Xác suất (%)": probs * 100
        })
        st.bar_chart(prob_df.set_index("Trạng thái"))
    else:
        st.info("👈 Hãy điều chỉnh các thông số bên trái và bấm nút **Dự đoán trạng thái rủi ro** để xem kết quả.")
