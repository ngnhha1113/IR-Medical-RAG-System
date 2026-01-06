import chromadb
import os
<<<<<<< HEAD
from tqdm import tqdm

class VectorDB:
    def __init__(self, collection_name="medical_cases"):
        # Persistent storage path
        self.db_path = "./chroma_db"
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_dataset(self, embedder, root_folder):
        """
        Recursively scans the root_folder and indexes images.
        Labels are extracted from the folder path (e.g., .../COVID19/...).
        """
        print(f"\n--- [VectorDB] Starting Indexing Process ---")
        print(f"Target Folder: {os.path.abspath(root_folder)}")
        
        if not os.path.exists(root_folder):
            print(f"❌ ERROR: Folder '{root_folder}' not found!")
            return

        # 1. Collect all image paths
        image_paths = []
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    image_paths.append(full_path)
        
        if len(image_paths) == 0:
            print("❌ No images found! Check your folder structure.")
            return

        print(f"✅ Found {len(image_paths)} images. Processing...")

        # 2. Indexing Loop
        existing_ids = set(self.collection.get()['ids'])
        new_count = 0
        
        for img_path in tqdm(image_paths):
            # Create unique ID based on relative path
            # e.g., "train_COVID19_img001.jpg"
            rel_path = os.path.relpath(img_path, root_folder)
            img_id = rel_path.replace(os.sep, "_")
            
            if img_id in existing_ids:
                continue

            try:
                # --- Logic to detect Label from Path ---
                path_upper = img_path.upper()
                
                diagnosis = "Normal" # Default
                guideline = "Routine checkup."

                if "COVID" in path_upper:
                    diagnosis = "COVID-19"
                    guideline = "Isolate patient. PCR Test. Monitor SPO2."
                elif "PNEUMONIA" in path_upper: 
                    diagnosis = "Pneumonia"
                    guideline = "Antibiotics (if bacterial). Chest X-Ray follow up."
                elif "NORMAL" in path_upper:
                    diagnosis = "Normal"
                    guideline = "No abnormalities detected."

                # Create Embedding
                embedding = embedder.encode_image(img_path)
                
                metadata = {
                    "filename": os.path.basename(img_path),
                    "path": img_path,
                    "diagnosis": diagnosis,
                    "guideline": guideline
                }

                self.collection.add(
                    ids=[img_id],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
                new_count += 1
            except Exception as e:
                print(f"⚠️ Error indexing {img_path}: {e}")

        print(f"✅ Indexing Finished! Added {new_count} new images.")
        print(f"📊 Total in Database: {self.collection.count()} items.")

    def search(self, query_vector, top_k=3):
        """Search function"""
        # Ensure DB is not empty before searching
        if self.collection.count() == 0:
            return None

=======
import pandas as pd
from tqdm import tqdm

class VectorDB:
    def __init__(self, collection_name="medical_cases_nih"):
        # Đường dẫn lưu DB
        self.db_path = "./chroma_db_nih"
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def index_dataset(self, embedder, csv_path, image_folder):
        """Hàm nạp dữ liệu từ CSV và ảnh vào ChromaDB"""
        print(f"\n--- [VectorDB] Bắt đầu Index dữ liệu ---")
        
        # Kiểm tra xem Database đã có dữ liệu chưa để tránh trùng
        if self.collection.count() > 0:
            print(f"⚠️ Database đã có {self.collection.count()} bản ghi.")
            print("👉 Nếu muốn cập nhật lại cột Giới Tính, hãy XÓA folder 'chroma_db_nih' rồi chạy lại!")
            return 

        print("📖 Đang đọc file CSV...")
        df = pd.read_csv(csv_path)
        
        # 1. Xóa khoảng trắng thừa ở tên cột
        df.columns = df.columns.str.strip()
        
        # 2. Đổi tên cột cho chuẩn (Mapping) - Đã bao gồm fix lỗi Patient Sex
        column_mapping = {
            'Age': 'Patient Age',
            'Gender': 'Patient Gender',
            'Sex': 'Patient Gender',         
            'Patient Sex': 'Patient Gender', 
            'View': 'View Position'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Kiểm tra cột
        if 'Patient Gender' not in df.columns:
            print(f"❌ VẪN LỖI: Không tìm thấy cột giới tính. Các cột hiện có: {df.columns.tolist()}")
            return

        available_images = set(os.listdir(image_folder))
        df_filtered = df[df['Image Index'].isin(available_images)]
        
        print(f"✅ Tìm thấy {len(df_filtered)} ảnh hợp lệ. Đang nạp vào DB...")

        count = 0
        batch_ids, batch_embeddings, batch_metadatas = [], [], []

        for index, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0]):
            filename = row['Image Index']
            img_path = os.path.join(image_folder, filename)
            
            try:
                # Lấy metadata
                gender = str(row['Patient Gender']) 
                age = str(row.get('Patient Age', 'N/A'))
                diagnosis = row.get('Finding Labels', 'Unknown')
                
                guideline = "Theo dõi định kỳ." if diagnosis == "No Finding" else "Cần hội chẩn chuyên khoa."

                metadata = {
                    "filename": filename,
                    "path": img_path,
                    "diagnosis": diagnosis,
                    "guideline": guideline,
                    "age": age,
                    "gender": gender
                }

                embedding = embedder.encode_image(img_path)
                
                batch_ids.append(filename)
                batch_embeddings.append(embedding)
                batch_metadatas.append(metadata)
                count += 1
                
                # Batch add (xử lý từng cụm 50 ảnh)
                if len(batch_ids) >= 50:
                    self.collection.add(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas)
                    batch_ids, batch_embeddings, batch_metadatas = [], [], []

            except Exception as e:
                pass

        # Add nốt số lẻ còn lại
        if batch_ids:
            self.collection.add(ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadatas)

        print(f"🎉 Hoàn tất! Đã Index {count} ảnh.")

    # --- ĐÂY LÀ HÀM SEARCH BẠN ĐANG BỊ THIẾU ---
    def search(self, query_vector, top_k=3):
        """Hàm tìm kiếm vector tương đồng"""
        if self.collection.count() == 0:
            return None
        
>>>>>>> b37cd91 (Upload medical retrieval project)
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        return results