import streamlit as st
<<<<<<< HEAD
from PIL import Image
import os

# Import backend modules
from backend.embedding import MedicalEmbedder
from backend.vector_db import VectorDB

# Page Configuration
st.set_page_config(page_title="Med-X Retrieval", layout="wide")

# --- CONFIGURATION ---
# POINT THIS TO YOUR ACTUAL DATA FOLDER
DATASET_PATH = "Data/train" 
=======
import os
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from streamlit_cropper import st_cropper 

# Import modules cũ
from backend.embedding import MedicalEmbedder
from backend.vector_db import VectorDB
from backend.utils import generate_heatmap_occlusion, zero_shot_classification
# Import module mới
from backend.atlas_utils import visualize_3d_atlas

# --- CẤU HÌNH ---
IMAGE_FOLDER_PATH = r"E:\IR\CXR8\images\Data\nih_dataset\images\images"
CSV_PATH = r"E:\IR\CXR8\images\Data\nih_dataset\Data_Entry_2017_v2020.csv"

# Cấu hình hiển thị
INITIAL_SHOW = 8
MAX_FETCH = 80

st.set_page_config(page_title="Med-X Pro Universe", layout="wide")
>>>>>>> b37cd91 (Upload medical retrieval project)

@st.cache_resource
def load_resources():
    embedder = MedicalEmbedder()
    db = VectorDB()
<<<<<<< HEAD
    
    # Check if dataset exists and index it
    if os.path.exists(DATASET_PATH):
        db.index_dataset(embedder, DATASET_PATH)
    else:
        print(f"⚠️ WARNING: Dataset path '{DATASET_PATH}' not found.")
        
    return embedder, db

def main():
    st.title("🩻 Med-X: Medical Image Retrieval System")
    st.caption(f"AI Assistant linked to dataset: {DATASET_PATH}")

    # Load AI
    with st.spinner("Initializing AI System..."):
        embedder, db = load_resources()

    col1, col2 = st.columns([1, 2])

    # --- Left: Input ---
    with col1:
        st.subheader("1. Input Image")
        uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])
        
        if uploaded_file:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.image(uploaded_file, caption="Query Image", use_column_width=True)
            
            if st.button("🔍 Analyze & Search", type="primary"):
                with st.spinner("Searching database..."):
                    query_vector = embedder.encode_image(temp_path)
                    results = db.search(query_vector, top_k=4)
                    st.session_state['search_results'] = results
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # --- Right: Results ---
    with col2:
        st.subheader("2. Retrieval Results")
        
        if 'search_results' in st.session_state:
            results = st.session_state['search_results']
            
            # Check if results are empty or None
            if results is None or not results['ids'][0]:
                st.error("❌ No results found in Database!")
                st.warning("Tip: Check if the 'Data/train' folder exists and is not empty.")
            else:
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]

                for i in range(0, len(metadatas), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(metadatas):
                            item = metadatas[i+j]
                            score = 1 - distances[i+j]
                            
                            with cols[j]:
                                # Display image
                                if os.path.exists(item['path']):
                                    st.image(item['path'], width=200)
                                else:
                                    st.image("https://via.placeholder.com/200?text=Image+Not+Found", width=200)
                                
                                st.markdown(f"**Case #{i+j+1}** (Sim: {score:.1%})")
                                
                                # Color code the diagnosis
                                diag = item['diagnosis']
                                color = "green" if "Normal" in diag else "red"
                                st.markdown(f":{color}[**{diag}**]")
                                
                                with st.expander("Details"):
                                    st.write(f"**File:** {item['filename']}")
                                    st.write(f"**Guideline:** {item['guideline']}")
        else:
            st.info("👈 Upload an image to start.")
=======
    if os.path.exists(IMAGE_FOLDER_PATH) and os.path.exists(CSV_PATH):
        db.index_dataset(embedder, CSV_PATH, IMAGE_FOLDER_PATH)
    return embedder, db

def display_results_grid(results):
    if results and results['ids'][0]:
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        cols_per_row = 4
        for i in range(0, len(metadatas), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(metadatas):
                    item = metadatas[i+j]
                    score = 1 - distances[i+j]
                    with cols[j]:
                        if os.path.exists(item['path']):
                            st.image(item['path'], width=150, caption=f"Rank #{i+j+1}")
                        with st.expander("🔎 Chi tiết"):
                            st.write(f"**Độ giống:** {score:.1%}")
                            st.write(f"**Chẩn đoán:** :red[{item['diagnosis']}]")
                            if os.path.exists(item['path']):
                                st.image(item['path'], use_container_width=True)
                            st.info(f"💡 {item['guideline']}")
    else:
        st.warning("Không tìm thấy kết quả.")

def visualize_statistics(results):
    # (Giữ nguyên hàm này từ code trước)
    if not results or not results['metadatas'][0]: return
    metadatas = results['metadatas'][0]
    total_cases = len(metadatas)
    all_diseases = []
    genders = []
    for meta in metadatas:
        diseases = meta['diagnosis'].split('|')
        all_diseases.extend(diseases)
        genders.append(meta['gender'])
    disease_counts = Counter(all_diseases)
    df_disease = pd.DataFrame.from_dict(disease_counts, orient='index', columns=['count']).sort_values(by='count', ascending=False)
    df_disease['percentage'] = (df_disease['count'] / total_cases) * 100
    
    st.markdown("### 📊 Dashboard Phân tích Lâm sàng")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("1. Dự báo nguy cơ")
        st.bar_chart(df_disease['count'], color="#FF4B4B")
        if not df_disease.empty:
            st.success(f"📌 **{df_disease.iloc[0]['percentage']:.1f}%** ca tương đồng **{df_disease.index[0]}**.")
    with col2:
        st.subheader("2. Giới tính")
        gender_counts = Counter(genders)
        st.dataframe(pd.DataFrame.from_dict(gender_counts, orient='index', columns=['count']), use_container_width=True)

def main():
    st.title("🩻 Med-X Pro: Medical Universe")
    
    if 'visible_count' not in st.session_state: st.session_state.visible_count = INITIAL_SHOW

    with st.spinner("Đang khởi động hệ thống AI..."):
        embedder, db = load_resources()

    with st.sidebar:
        st.header("Input Data")
        uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            input_image = Image.open(uploaded_file).convert("RGB")
            st.image(input_image, caption="Ảnh gốc", use_container_width=True)
            
            st.divider()
            st.subheader("🤖 AI Scan Nhanh")
            candidate_labels = ["Normal", "Pneumonia", "Cardiomegaly", "Effusion", "Nodule", "Atelectasis", "Mass"]
            if st.button("⚡ Quét dấu hiệu"):
                with st.spinner("Đang quét..."):
                    scores = zero_shot_classification(embedder, input_image, candidate_labels)
                for label, score in scores:
                    color = ":red" if score > 0.5 else ":orange" if score > 0.2 else ":green"
                    st.markdown(f"**{label}**: {color}[{score:.1%}]")
                    st.progress(score)
        else:
            input_image = None

    # --- THÊM TAB 5: VŨ TRỤ BỆNH LÝ ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Tìm kiếm", "➕ Ảnh + Text", "✂️ ROI", "🔥 Heatmap", "🌌 Spatial Analysis"
    ])

    def run_search(vector):
        st.session_state.visible_count = INITIAL_SHOW
        results = db.search(vector, top_k=MAX_FETCH)
        st.session_state['full_results'] = results

    with tab1:
        st.subheader("Tìm kiếm tương đồng")
        if input_image and st.button("🚀 Tìm kiếm", key="btn1"):
            vec = embedder.encode_image(input_image)
            run_search(vec)

    with tab2:
        st.subheader("Điều chỉnh vector")
        if input_image:
            c1, c2 = st.columns([3, 1])
            with c1: tm = st.text_input("Thêm text", key="tm")
            with c2: w = st.slider("Trọng số", 0.1, 2.0, 0.5, key="wm")
            if st.button("🚀 Tìm kết hợp", key="btn2") and tm:
                iv = np.array(embedder.encode_image(input_image))
                tv = np.array(embedder.encode_text(tm))
                run_search(((iv + (tv * w)) / np.linalg.norm(iv + (tv * w))).tolist())

    with tab3:
        st.subheader("Tìm theo vùng cắt")
        if input_image:
            c1, c2 = st.columns([3, 1])
            with c1:
                z = st.slider("Zoom", 400, 1500, 700, 50, key="zr")
                wp = z / float(input_image.size[0])
                crop = st_cropper(input_image.resize((z, int(input_image.size[1]*wp)), Image.Resampling.LANCZOS), realtime_update=True, box_color='#FF0000', aspect_ratio=None)
            with c2:
                if crop:
                    st.image(crop, width=150)
                    if st.button("🚀 Tìm vùng", key="btn3"):
                        run_search(embedder.encode_image(crop))

    with tab4:
        st.subheader("🔥 Heatmap")
        if input_image and st.button("⚡ Tạo Heatmap"):
            with st.spinner("Đang tính toán..."):
                st.image(generate_heatmap_occlusion(embedder, input_image), caption="AI Attention Map", use_container_width=True)

    # === TAB 5: VŨ TRỤ BỆNH LÝ (MỚI) ===
    with tab5:
        st.subheader("🌌 3D Interactive Vector Atlas")
        st.markdown("""
        **Hướng dẫn:** - Các điểm tròn: Dữ liệu bệnh án trong Database.
        - **Hình Thoi ĐỎ (♦)**: Vị trí ảnh bạn vừa upload.
        - Xoay chuột để xem ảnh của bạn đang "đứng" gần cụm bệnh nào nhất.
        """)
        
        # Nút tạo vũ trụ
        if st.button("🪐 Khởi tạo Vũ trụ 3D"):
            with st.spinner("Đang tính toán PCA không gian 3 chiều..."):
                # Nếu có ảnh input, lấy vector để vẽ điểm đỏ
                if input_image:
                    input_vec = embedder.encode_image(input_image)
                    fig = visualize_3d_atlas(db, input_image_vector=input_vec)
                else:
                    fig = visualize_3d_atlas(db, input_image_vector=None)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Chưa có đủ dữ liệu trong Database để vẽ biểu đồ.")

    # --- KẾT QUẢ ---
    st.divider()
    if 'full_results' in st.session_state:
        full = st.session_state['full_results']
        if full and full['ids'][0]:
            limit = st.session_state.visible_count
            sliced = {k: [v[0][:limit]] for k, v in full.items() if k in ['ids', 'distances', 'metadatas']}
            
            st.subheader(f"📋 Kết quả ({len(sliced['ids'][0])} ảnh)")
            display_results_grid(sliced)
            
            if limit < len(full['ids'][0]):
                if st.button("⬇️ Xem thêm"):
                    st.session_state.visible_count += 8
                    st.rerun()

            st.divider()
            visualize_statistics(full)
>>>>>>> b37cd91 (Upload medical retrieval project)

if __name__ == "__main__":
    main()