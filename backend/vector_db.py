import os

import chromadb
import pandas as pd
from tqdm import tqdm


class VectorDB:
    def __init__(self, collection_name="medical_cases_nih"):
        self.db_path = "./chroma_db_nih"
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_dataset(self, embedder, csv_path, image_folder):
        print(f"\n--- [VectorDB] Bắt đầu Index dữ liệu ---")

        # Kiểm tra nếu DB đã có dữ liệu thì thôi
        if self.collection.count() > 0:
            print(f"⚠️ Database đã có {self.collection.count()} bản ghi.")
            print(
                "👉 Nếu muốn cập nhật lại, hãy XÓA folder 'chroma_db_nih' rồi chạy lại!"
            )
            return

        print("📖 Đang đọc file CSV...")
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Mapping cột (bao gồm fix lỗi Patient Sex)
        column_mapping = {
            "Age": "Patient Age",
            "Gender": "Patient Gender",
            "Sex": "Patient Gender",
            "Patient Sex": "Patient Gender",
            "View": "View Position",
        }
        df.rename(columns=column_mapping, inplace=True)

        if "Patient Gender" not in df.columns:
            print(
                f"❌ VẪN LỖI: Không tìm thấy cột giới tính. Columns: {df.columns.tolist()}"
            )
            return

        available_images = set(os.listdir(image_folder))
        df_filtered = df[df["Image Index"].isin(available_images)]
        print(f"✅ Tìm thấy {len(df_filtered)} ảnh hợp lệ. Đang nạp vào DB...")

        count = 0
        batch_ids, batch_embeddings, batch_metadatas = [], [], []

        for index, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0]):
            filename = row["Image Index"]
            img_path = os.path.join(image_folder, filename)

            try:
                gender = str(row["Patient Gender"])
                age = str(row.get("Patient Age", "N/A"))
                diagnosis = row.get("Finding Labels", "Unknown")
                guideline = (
                    "Theo dõi định kỳ."
                    if diagnosis == "No Finding"
                    else "Cần hội chẩn chuyên khoa."
                )

                metadata = {
                    "filename": filename,
                    "path": img_path,
                    "diagnosis": diagnosis,
                    "guideline": guideline,
                    "age": age,
                    "gender": gender,
                }

                embedding = embedder.encode_image(img_path)

                batch_ids.append(filename)
                batch_embeddings.append(embedding)
                batch_metadatas.append(metadata)
                count += 1

                if len(batch_ids) >= 50:
                    self.collection.add(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        metadatas=batch_metadatas,
                    )
                    batch_ids, batch_embeddings, batch_metadatas = [], [], []

            except Exception as e:
                pass

        if batch_ids:
            self.collection.add(
                ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas
            )

        print(f"🎉 Hoàn tất! Đã Index {count} ảnh.")

    # --- 👇 QUAN TRỌNG: HÀM SEARCH PHẢI NẰM Ở ĐÂY 👇 ---
    def search(self, query_vector, top_k=3):
        if self.collection.count() == 0:
            return None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["metadatas", "distances", "embeddings"],
        )
        return results
