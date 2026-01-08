import cv2
import numpy as np
import torch
from PIL import Image


# --- HÀM 1: HEATMAP (GIỮ NGUYÊN) ---
def generate_heatmap_occlusion(embedder, image, step=32):
    original_vec = np.array(embedder.encode_image(image))

    small_size = (224, 224)
    img_pil_small = image.resize(small_size)
    img_np = np.array(img_pil_small)

    h, w, _ = img_np.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, h, step):
            for x in range(0, w, step):
                masked_img = img_np.copy()
                masked_img[y : y + step, x : x + step, :] = 0
                masked_pil = Image.fromarray(masked_img)
                masked_vec = np.array(embedder.encode_image(masked_pil))
                dist = np.linalg.norm(original_vec - masked_vec)
                heatmap[y : y + step, x : x + step] = dist

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_resized = cv2.resize(heatmap, image.size)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    original_np = np.array(image)
    superimposed = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)

    return Image.fromarray(superimposed)


# --- HÀM 2: ZERO-SHOT CLASSIFICATION (MỚI) ---
def zero_shot_classification(embedder, image, labels):
    """
    So sánh ảnh với danh sách bệnh (text) để chấm điểm tương đồng.
    Không cần train model, dùng kiến thức có sẵn của CLIP/BiomedCLIP.
    """
    device = embedder.device

    # 1. Tạo câu prompt giả
    text_prompts = [f"A chest x-ray showing {label}" for label in labels]

    # 2. Tokenize Text
    text_inputs = embedder.tokenizer(text_prompts).to(device)

    # 3. Xử lý ảnh (Cần preprocess lại từ đầu để lấy Tensor)
    image_input = embedder.preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # Lấy feature từ model gốc
        image_features = embedder.model.encode_image(image_input)
        text_features = embedder.model.encode_text(text_inputs)

        # Chuẩn hóa vector
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Tính độ tương đồng (Cosine Similarity)
        # Nhân ma trận: (1, 512) x (N_labels, 512)^T -> (1, N_labels)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Lấy kết quả
        values, indices = similarity[0].topk(len(labels))

    results = []
    for value, index in zip(values, indices):
        results.append((labels[index], value.item()))

    return results


def rocchio_update(
    original_query,
    relevant_vectors,
    non_relevant_vectors,
    alpha=1.0,
    beta=0.75,
    gamma=0.15,
):
    """
    Rocchio algorithm for relevance feedback.
    q_new = alpha * q_old + beta * mean(relevant) - gamma * mean(non_relevant)
    """
    q_old = np.array(original_query)

    if not relevant_vectors:
        vec_relevant = np.zeros_like(q_old)
    else:
        vec_relevant = np.mean(relevant_vectors, axis=0)

    if not non_relevant_vectors:
        vec_non_relevant = np.zeros_like(q_old)
    else:
        vec_non_relevant = np.mean(non_relevant_vectors, axis=0)

    q_new = alpha * q_old + beta * vec_relevant - gamma * vec_non_relevant

    # Normalize
    norm = np.linalg.norm(q_new)
    if norm > 0:
        q_new /= norm

    return q_new.tolist()
