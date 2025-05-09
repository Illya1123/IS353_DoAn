import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from skimage.segmentation import slic
from skimage.color import rgb2lab, rgb2gray
from skimage.measure import regionprops
import os
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool
from torch.nn import Linear, ReLU, Dropout
import networkx as nx

# === Config ===
st.set_page_config(page_title="SLIC", layout="centered")
st.title("🌍 Scene Classification with GCN + Superpixels")

# === Label mapping ===
code = {'buildings': 0, 'forest': 1, 'glacier': 2, 'mountain': 3, 'sea': 4, 'street': 5}
inv_code = {v: k for k, v in code.items()}
new_size = 100

# === Model class ===
class GCNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dims[0])
        self.bn1 = BatchNorm(hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.bn2 = BatchNorm(hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], hidden_dims[2])
        self.bn3 = BatchNorm(hidden_dims[2])
        self.fc1 = Linear(hidden_dims[2], hidden_dims[3])
        self.fc2 = Linear(hidden_dims[3], num_classes)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.relu(self.bn2(self.conv2(x, edge_index)))
        x = self.relu(self.bn3(self.conv3(x, edge_index)))
        x = global_mean_pool(x, batch)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# === Graph processing ===
def compute_rag(image, segments):
    graph = nx.Graph()
    lab_image = rgb2lab(image)
    for i in range(segments.shape[0]):
        for j in range(segments.shape[1]):
            current = segments[i, j]
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < segments.shape[0] and 0 <= nj < segments.shape[1]:
                    neighbor = segments[ni, nj]
                    if neighbor != current:
                        graph.add_edge(current, neighbor)
    for node in graph.nodes():
        mask = (segments == node)
        mean_color = np.mean(lab_image[mask], axis=0)
        graph.nodes[node]['feature'] = mean_color
    return graph

def extract_superpixel_features(image, segments):
    grayscale_image = rgb2gray(image)
    features = []
    props = regionprops(segments, intensity_image=grayscale_image)
    for region in props:
        mask = (segments == region.label)
        mean_color = np.mean(image[mask], axis=0)
        eccentricity = region.eccentricity
        bbox = region.bbox
        aspect_ratio = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0]) if (bbox[2] - bbox[0]) > 0 else 0
        solidity = region.solidity
        centroid = region.centroid
        perimeter = region.perimeter
        features.append([*mean_color, eccentricity, aspect_ratio, solidity, *centroid, perimeter])
    return np.array(features)

def image_to_graph(image, n_segments=400):
    segments = slic(image, n_segments=n_segments, compactness=10)
    graph = compute_rag(image, segments)
    features = extract_superpixel_features(image, segments)
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(graph.nodes())}
    mapped_edges = [(node_mapping[u], node_mapping[v]) for u, v in graph.edges if u in node_mapping and v in node_mapping]
    edge_index = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GCNModel(input_dim=9, hidden_dims=[128, 64, 32, 16], num_classes=6)
model_path = "model/gcn_slic.pth"

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
else:
    st.error("⚠️ Model file not found! Please train and save it as `model/gcn_slic.pth`.")
    st.stop()

# === Inference ===
def predict_image(image_np):
    image_resized = cv2.resize(image_np, (new_size, new_size))
    graph = image_to_graph(image_resized)
    graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long)
    graph = graph.to(device)

    with torch.no_grad():
        out = model(graph)
        pred = out.argmax().item()
    return pred

# === Input UI ===
st.sidebar.header("🖼️ Upload Image or URL")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
url_input = st.sidebar.text_input("Or paste image URL")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
elif url_input:
    try:
        response = requests.get(url_input)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    except:
        st.error("❌ Failed to load image from URL.")
        st.stop()
else:
    st.info("Please upload an image or provide an image URL.")
    st.stop()

# === Main Prediction ===
image_np = np.array(image)
st.image(image_np, caption="Input Image", use_container_width=True)

with st.spinner("🔍 Analyzing with GCN..."):
    pred_class = predict_image(image_np)

st.success(f"🎯 **Predicted Scene**: `{inv_code[pred_class]}`")
