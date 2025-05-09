# 🧠 Intel Image Classification - Scene Recognition

## 📌 Giới thiệu
Dự án này tập trung vào bài toán phân loại ảnh cảnh vật từ dataset **Intel Image Classification**. Mục tiêu là xây dựng mô hình học sâu (deep learning) để nhận dạng chính xác các loại cảnh như: rừng, biển, đô thị, núi, băng tuyết,...

---

## 👨‍💻 Thành viên nhóm

| Họ tên           | MSSV         | Vai trò              |
|------------------|--------------|-----------------------|
| Lê Quốc Anh     | 21520565      | Xử lý dữ liệu, mô hình, huấn luyện & visualization |
| Nguyễn Hoàng Quý       | 21520425      | Đưa ý tưởng, viết báo cáo |
| Cao Mỹ Duyên         | 22520347      | Nghiên cứu tài liệu, viết báo cáo |
| Nguyễn Thiên Kim       | 22520729      | Nghiên cứu tài liệu, viết báo cáo |

---

## 📂 Dataset

- **Nguồn:** [Intel Image Classification on Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Các lớp (classes):**
  - Buildings
  - Forest
  - Glacier
  - Mountain
  - Sea
  - Street


## 👨‍💻 Cách chạy code

```bash
# Bước 1: Di chuyển vào thư mục dự án
cd IS353_DoAn

# Bước 2: Tạo virtual environment (venv)
python -m venv venv

# Bước 3: Kích hoạt virtual environment
# Windows
venv/Scripts/activate

# Bước 4: Cài đặt thư viện từ file requirement.txt
pip install -r requirement.txt

# Bước 5: Chạy ứng dụng Streamlit
streamlit run app.py (Đối với test model gcn-dt và gcn-combine)
hoặc
streamlit run slic.py (Đối với gcn-slic)
```

> 🔁 **Lưu ý:** Mỗi lần mở terminal mới, bạn cần **kích hoạt lại virtual environment** bằng lệnh `venv\Scripts\activate` (Windows) hoặc `source venv/bin/activate` (Linux/macOS) trước khi chạy Streamlit.
