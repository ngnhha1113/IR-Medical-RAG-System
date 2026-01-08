import os
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

from backend.atlas_utils import visualize_3d_atlas
from backend.embedding import MedicalEmbedder
from backend.utils import (
    generate_heatmap_occlusion,
    rocchio_update,
    zero_shot_classification,
)
from backend.vector_db import VectorDB

# --- CẤU HÌNH ---
IMAGE_FOLDER_PATH = r"Data/train"
CSV_PATH = r"Data_Entry_2017_v2020.csv"

# Cấu hình hiển thị
INITIAL_SHOW = 8
MAX_FETCH = 80

st.set_page_config(page_title="Med-X Pro Universe", layout="wide")


@st.cache_resource
def load_resources():
    embedder = MedicalEmbedder()
    db = VectorDB()
    if os.path.exists(IMAGE_FOLDER_PATH) and os.path.exists(CSV_PATH):
        db.index_dataset(embedder, CSV_PATH, IMAGE_FOLDER_PATH)
    return embedder, db


def display_results_grid(results):
    if results and results["ids"][0]:
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]
        cols_per_row = 4
        for i in range(0, len(metadatas), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < len(metadatas):
                    item = metadatas[i + j]
                    score = 1 - distances[i + j]
                    doc_id = ids[i + j]
                    with cols[j]:
                        if os.path.exists(item["path"]):
                            st.image(
                                item["path"], width=150, caption=f"Rank #{i + j + 1}"
                            )

                        # Relevance Checkbox
                        st.checkbox("Relevant", key=f"rel_{doc_id}")

                        with st.expander("🔎 Chi tiết"):
                            st.write(f"**Độ giống:** {score:.1%}")
                            st.write(f"**Chẩn đoán:** :red[{item['diagnosis']}]")
                            if os.path.exists(item["path"]):
                                st.image(item["path"], width="stretch")
                            st.info(f"💡 {item['guideline']}")
    else:
        st.warning("Không tìm thấy kết quả.")


def visualize_statistics(results):
    # (Giữ nguyên hàm này từ code trước)
    if not results or not results["metadatas"][0]:
        return
    metadatas = results["metadatas"][0]
    total_cases = len(metadatas)
    all_diseases = []
    genders = []
    for meta in metadatas:
        diseases = meta["diagnosis"].split("|")
        all_diseases.extend(diseases)
        genders.append(meta["gender"])
    disease_counts = Counter(all_diseases)
    df_disease = pd.DataFrame.from_dict(
        disease_counts, orient="index", columns=["count"]
    ).sort_values(by="count", ascending=False)
    df_disease["percentage"] = (df_disease["count"] / total_cases) * 100

    st.markdown("### 📊 Dashboard Phân tích Lâm sàng")
    st.subheader("Giới tính")
    gender_counts = Counter(genders)
    st.dataframe(
        pd.DataFrame.from_dict(gender_counts, orient="index", columns=["count"]),
        width="stretch",
    )


def main():
    st.title("🩻 Med-X Pro: Medical Universe")

    if "visible_count" not in st.session_state:
        st.session_state.visible_count = INITIAL_SHOW

    with st.spinner("Đang khởi động hệ thống AI..."):
        embedder, db = load_resources()

    def run_search(vector):
        st.session_state.visible_count = INITIAL_SHOW
        st.session_state["current_query_vector"] = vector
        results = db.search(vector, top_k=MAX_FETCH)
        st.session_state["full_results"] = results

    with st.sidebar:
        st.header("Input Data")
        uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png", "jpeg"])
        input_image = None

        if uploaded_file:
            input_image = Image.open(uploaded_file).convert("RGB")
            st.image(input_image, caption="Ảnh gốc", width="stretch")

            st.divider()
            st.subheader("🤖 AI Scan Nhanh")
            candidate_labels = [
                "Normal",
                "Pneumonia",
                "Cardiomegaly",
                "Effusion",
                "Nodule",
                "Atelectasis",
                "Mass",
            ]
            if st.button("⚡ Quét dấu hiệu"):
                with st.spinner("Đang quét..."):
                    scores = zero_shot_classification(
                        embedder, input_image, candidate_labels
                    )
                for label, score in scores:
                    color = (
                        ":red"
                        if score > 0.5
                        else ":orange"
                        if score > 0.2
                        else ":green"
                    )
                    st.markdown(f"**{label}**: {color}[{score:.1%}]")
                    st.progress(score)

        st.divider()
        st.subheader("🎯 Relevance Feedback")
        alpha = st.slider(
            "Alpha (Original)", 0.0, 2.0, 1.0, help="Weight for original query"
        )
        beta = st.slider(
            "Beta (Relevant)", 0.0, 2.0, 0.75, help="Weight for relevant docs"
        )
        gamma = st.slider(
            "Gamma (Non-relevant)", 0.0, 2.0, 0.15, help="Weight for non-relevant docs"
        )

        if st.button("🔄 Refine Search"):
            if (
                "full_results" in st.session_state
                and "current_query_vector" in st.session_state
            ):
                full = st.session_state["full_results"]
                if full and "embeddings" in full:
                    ids = full["ids"][0]
                    embeddings = full["embeddings"][0]

                    relevant_vecs = []
                    non_relevant_vecs = []

                    for idx, doc_id in enumerate(ids):
                        # Check checkbox state
                        if st.session_state.get(f"rel_{doc_id}", False):
                            relevant_vecs.append(embeddings[idx])
                        else:
                            non_relevant_vecs.append(embeddings[idx])

                    if not relevant_vecs:
                        st.warning("Please select at least one relevant result.")
                    else:
                        new_query = rocchio_update(
                            st.session_state["current_query_vector"],
                            relevant_vecs,
                            non_relevant_vecs,
                            alpha,
                            beta,
                            gamma,
                        )
                        run_search(new_query)
                        st.rerun()
                else:
                    st.error("No search results to refine.")

    # --- THÊM TAB 5: VŨ TRỤ BỆNH LÝ ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🔍 Tìm kiếm", "➕ Ảnh + Text", "✂️ ROI", "🔥 Heatmap", "🌌 Vũ trụ Bệnh lý"]
    )

    with tab1:
        st.subheader("Tìm kiếm tương đồng")
        if input_image and st.button("🚀 Tìm kiếm", key="btn1"):
            vec = embedder.encode_image(input_image)
            run_search(vec)

    with tab2:
        st.subheader("Điều chỉnh vector")
        if input_image:
            c1, c2 = st.columns([3, 1])
            with c1:
                tm = st.text_input("Thêm text", key="tm")
            with c2:
                w = st.slider("Trọng số", 0.1, 2.0, 0.5, key="wm")
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
                crop = st_cropper(
                    input_image.resize(
                        (z, int(input_image.size[1] * wp)), Image.Resampling.LANCZOS
                    ),
                    realtime_update=True,
                    box_color="#FF0000",
                    aspect_ratio=None,
                )
            with c2:
                if crop:
                    st.image(crop, width=150)
                    if st.button("🚀 Tìm vùng", key="btn3"):
                        run_search(embedder.encode_image(crop))

    with tab4:
        st.subheader("🔥 Heatmap")
        if input_image and st.button("⚡ Tạo Heatmap"):
            with st.spinner("Đang tính toán..."):
                st.image(
                    generate_heatmap_occlusion(embedder, input_image),
                    caption="AI Attention Map",
                    width="stretch",
                )

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
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.error("Chưa có đủ dữ liệu trong Database để vẽ biểu đồ.")

    # --- KẾT QUẢ ---
    st.divider()
    if "full_results" in st.session_state:
        full = st.session_state["full_results"]
        if full and full["ids"][0]:
            limit = st.session_state.visible_count
            sliced = {
                k: [v[0][:limit]]
                for k, v in full.items()
                if k in ["ids", "distances", "metadatas"]
            }

            st.subheader(f"📋 Kết quả ({len(sliced['ids'][0])} ảnh)")
            display_results_grid(sliced)

            if limit < len(full["ids"][0]):
                if st.button("⬇️ Xem thêm"):
                    st.session_state.visible_count += 8
                    st.rerun()

            st.divider()
            visualize_statistics(full)


if __name__ == "__main__":
    main()
