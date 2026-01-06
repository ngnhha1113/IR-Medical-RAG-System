import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

def get_atlas_data(db, limit=1500):
    """
    Lấy mẫu ngẫu nhiên từ ChromaDB để xây dựng 'Vũ trụ'.
    """
    # Lấy dữ liệu thô từ DB
    data = db.collection.get(
        include=["embeddings", "metadatas"],
        limit=limit
    )
    
    # --- SỬA LỖI VALUE ERROR TẠI ĐÂY ---
    # Kiểm tra an toàn: Nếu không có embeddings hoặc danh sách rỗng
    if data['embeddings'] is None or len(data['embeddings']) == 0:
        return None, None, None # Trả về 3 giá trị None

    embeddings = np.array(data["embeddings"])
    metadatas = data["metadatas"]
    
    # Trích xuất nhãn bệnh chính (Primary Label) để tô màu
    labels = []
    filenames = []
    for meta in metadatas:
        # Lấy bệnh đầu tiên trong danh sách (VD: "Infiltration|Mass" -> "Infiltration")
        primary_disease = meta.get("diagnosis", "Unknown").split("|")[0]
        
        # Gom các bệnh phổ biến, còn lại đưa vào Others
        common_diseases = ["Pneumonia", "Infiltration", "Atelectasis", "Effusion", "Nodule", "No Finding", "Cardiomegaly", "Mass"]
        
        if primary_disease in common_diseases:
            labels.append(primary_disease)
        else:
            labels.append("Others")
            
        filenames.append(meta.get("filename", "Unknown"))

    return embeddings, labels, filenames

def visualize_3d_atlas(db, input_image_vector=None):
    """
    Vẽ biểu đồ 3D:
    - Các điểm mờ: Dữ liệu nền (Atlas).
    - Ngôi sao đỏ: Ảnh người dùng vừa upload.
    """
    # 1. Lấy dữ liệu nền
    embeddings, labels, filenames = get_atlas_data(db)
    
    # Nếu không lấy được dữ liệu (hoặc DB rỗng)
    if embeddings is None:
        return None

    # 2. Gộp vector input của user vào (nếu có) để tính toán không gian chung
    if input_image_vector is not None:
        input_vec = np.array(input_image_vector).reshape(1, -1)
        # Nối vector input vào cuối danh sách
        all_vectors = np.vstack([embeddings, input_vec])
        # Thêm nhãn cho điểm này
        labels.append("YOUR IMAGE") 
        filenames.append("Input Upload")
        is_user_input = [False] * len(embeddings) + [True] # Đánh dấu điểm cuối là User
    else:
        all_vectors = embeddings
        is_user_input = [False] * len(embeddings)

    # 3. Giảm chiều dữ liệu (512 -> 3 chiều) bằng PCA
    # Cần ít nhất 3 điểm dữ liệu để chạy PCA 3 chiều
    if len(all_vectors) < 3:
        return None

    pca = PCA(n_components=3)
    components = pca.fit_transform(all_vectors)

    # 4. Tạo DataFrame cho Plotly
    df = pd.DataFrame(components, columns=['x', 'y', 'z'])
    df['label'] = labels
    df['filename'] = filenames
    df['is_user'] = is_user_input
    
    # Kích thước điểm: Điểm của user to hơn (size 15), còn lại nhỏ (size 5)
    df['size'] = df['is_user'].apply(lambda x: 15 if x else 5)
    # Ký hiệu: User là hình thoi (diamond), còn lại là hình tròn (circle)
    df['symbol'] = df['is_user'].apply(lambda x: 'diamond' if x else 'circle')

    # 5. Vẽ biểu đồ tương tác
    fig = px.scatter_3d(
        df, x='x', y='y', z='z',
        color='label',
        symbol='symbol',
        size='size',
        hover_data=['filename', 'label'],
        opacity=0.7,
        title="Vũ trụ Bệnh lý (3D Vector Space)",
        color_discrete_map={"YOUR IMAGE": "#FF0000"} # Màu đỏ cho ảnh user
    )

    # Tinh chỉnh giao diện
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        scene=dict(
            xaxis_title='PCA 1',
            yaxis_title='PCA 2',
            zaxis_title='PCA 3'
        ),
        legend_title="Nhóm bệnh lý"
    )
    
    return fig