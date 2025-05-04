import streamlit as st
import torch
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image
import requests
import io
from torch_geometric.data import Data
from scipy.spatial import Delaunay
from skimage.transform import resize
from torch.nn import Linear, ReLU
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool

# ==== Label code ====
code = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
inv_code = {v: k for k, v in code.items()}

from skimage.segmentation import slic
from skimage.color import rgb2lab
from sklearn.preprocessing import StandardScaler

from skimage.measure import regionprops

def create_combined_graph(image, n_segments=100, compactness=10):
    # Resize ảnh
    image = resize(image, (100, 100), anti_aliasing=True)
    
    # SLIC Segmentation
    segments = slic(image, n_segments=n_segments, compactness=compactness, sigma=1)
    regions = regionprops(segments + 1)
    
    # Tính features cho superpixels
    centers = np.array([r.centroid for r in regions])  # Vị trí trung tâm
    colors = np.array([image[segments == i].mean(axis=0) for i in np.unique(segments)])  # Màu trung bình
    
    # Delaunay Triangulation
    tri = Delaunay(centers)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            edge = tuple(sorted((simplex[i], simplex[(i + 1) % 3])))
            edges.add(edge)
    edges = np.array(list(edges)).T
    
    # Tạo đồ thị PyTorch Geometric
    x = torch.tensor(np.hstack([colors, centers]), dtype=torch.float)  # Features: màu (3) + vị trí (2)
    edge_index = torch.tensor(edges, dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    
    return data, segments




# ==== Define Model ====
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.batch_norm1 = BatchNorm(hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.batch_norm2 = BatchNorm(hidden_dims[2])
        self.fc = Linear(hidden_dims[2], num_classes)
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.batch_norm1(x)
        x = self.relu(self.conv3(x, edge_index))
        x = self.batch_norm2(x)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x
    
class CombinedGCN(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dims=[256, 128, 64], num_classes=6):
        super(CombinedGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.bn1 = BatchNorm(hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.bn2 = BatchNorm(hidden_dims[2])
        self.fc = Linear(hidden_dims[2], num_classes)
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.bn1(x)
        x = self.relu(self.conv3(x, edge_index))
        x = self.bn2(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)


# ==== Image to Graph ====
def image_to_graph(image, new_size=(32, 32)):
    image = np.array(image)
    h, w, c = image.shape
    image = resize(image, new_size, anti_aliasing=True)
    pixels = image.reshape(-1, c)
    positions = np.column_stack(np.unravel_index(np.arange(new_size[0] * new_size[1]), (new_size[0], new_size[1])))
    tri = Delaunay(positions)
    edges = [(simplex[i], simplex[(i + 1) % 3]) for simplex in tri.simplices for i in range(3)]
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    x = torch.tensor(pixels, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(x.size(0), dtype=torch.long)  # Add batch info
    return data


# ==== Load Model ==== 
@st.cache_resource
def load_model(model_name):
    if model_name == 'gcn_dt':
        model = GCNModel(input_dim=3, hidden_dims=[256, 128, 64], num_classes=6)
    elif model_name == 'gcn_combine':
        model = CombinedGCN(input_dim=5, hidden_dims=[256, 128, 64], num_classes=6)
    else:
        raise ValueError("Tên mô hình không hợp lệ")
    
    model_path = f"model/{model_name}.pth"
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model




# ==== Prediction Function ====
def predict_image(img, model, model_name):
    image = np.array(img)

    if model_name == 'gcn_dt':
        graph = image_to_graph(img)  # Dựa vào pixel (3 features)
    elif model_name == 'gcn_combine':
        graph, _ = create_combined_graph(image)  # Dựa vào SLIC + Delaunay (5 features)
    else:
        raise ValueError("Tên mô hình không hợp lệ")

    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)  # cần cho global_mean_pool
    with torch.no_grad():
        out = model(graph)
        pred = out.argmax(dim=1).item()
    return inv_code[pred]



# ==== Streamlit UI ====
st.title("🌍 GCN Scene Classification")
st.write("Upload an image or paste an image URL to classify it using the trained GCN model.")

model_option = st.selectbox("Chọn mô hình:", ["gcn_dt", "gcn_combine"])
model = load_model(model_option)

option = st.radio("Chọn phương thức nhập ảnh:", ("📁 Upload từ máy", "🌐 Từ URL"))

img = None
if option == "📁 Upload từ máy":
    uploaded_file = st.file_uploader("Chọn ảnh", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
elif option == "🌐 Từ URL":
    url = st.text_input("Nhập URL ảnh:")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(io.BytesIO(response.content)).convert("RGB")
        except:
            st.error("Không thể tải ảnh từ URL. Vui lòng kiểm tra lại.")

if img:
    st.image(img, caption="Ảnh đầu vào", use_container_width=True)
    if st.button("🔍 Dự đoán"):
        with st.spinner("Đang dự đoán..."):
            prediction = predict_image(img, model, model_option)
            st.success(f"✅ Dự đoán: **{prediction.upper()}**")

