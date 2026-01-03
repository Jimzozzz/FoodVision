import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
import io
import pandas as pd

# -------------------- Page config --------------------
st.set_page_config(page_title="FoodVision Guard — Bread", layout="wide")

# -------------------- CSS (Dark UI) --------------------
st.markdown("""
<style>
/* Global font + dark background */
html, body, [class*="css"] {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Noto Sans Thai', sans-serif;
}

:root{
  --bg:#0b1220;         /* page background */
  --panel:#111827;      /* card background */
  --muted:#0f172a;      /* muted panels */
  --border:#263246;     /* border */
  --text:#e5e7eb;       /* text */
  --sub:#b6c0cf;        /* secondary text */
  --accent:#60a5fa;     /* accent */
}

/* Set Streamlit app background */
.stApp {
  background: radial-gradient(1200px 600px at 20% 0%, rgba(96,165,250,0.12), transparent 60%),
              radial-gradient(900px 500px at 80% 10%, rgba(34,197,94,0.10), transparent 55%),
              var(--bg);
  color: var(--text);
}

.block-container { padding-top: 16px; padding-bottom: 24px; max-width: 1200px; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* Text colors */
h1,h2,h3,h4,h5,h6, p, span, div, label { color: var(--text) !important; }
small, .stCaption, .stMarkdown p { color: var(--sub) !important; }

/* Card */
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 24px rgba(0,0,0,0.25);
}
.card .title { font-size: 22px; font-weight: 800; letter-spacing: 0.2px; }
.card .sub { margin-top: 6px; color: var(--sub) !important; font-size: 14px; }

/* Metrics look */
div[data-testid="stMetric"] {
  background: var(--muted);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 12px;
}

/* File uploader */
div[data-testid="stFileUploader"] section {
  background: var(--muted);
  border: 1px dashed rgba(99,102,241,0.55);
  border-radius: 14px;
}
div[data-testid="stFileUploader"] * { color: var(--text) !important; }

/* Buttons */
.stButton>button, .stDownloadButton>button {
  border-radius: 12px;
  border: 1px solid var(--border);
}

/* Selection highlight (ตอนลากเลือกข้อความ) */
::selection { background: rgba(96,165,250,0.45); color: var(--text); }
</style>
""", unsafe_allow_html=True)

# -------------------- Config --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "best_bread_mold.pt"

IMG_SIZE_DEFAULT = 384
THRESHOLD_DEFAULT = 0.60

# ⚠️ ต้องตรงกับตอน train
CLASSES = ["clean", "mold"]

# -------------------- Grad-CAM --------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.activations = None
        self.gradients = None

        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx):
        logits = self.model(x)
        self.model.zero_grad()
        logits[:, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam[0].cpu().numpy()

# -------------------- Utils --------------------
def disable_inplace_activations(model):
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.SiLU)):
            m.inplace = False

@st.cache_resource
def load_model(img_size):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    disable_inplace_activations(model)

    state = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    cam_engine = GradCAM(model, model.features[-1][0])

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    return model, cam_engine, tfm

def jet_colormap(x):
    """x in [0,1] -> RGB heatmap (jet-like) without matplotlib/opencv"""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4*x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4*x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4*x - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)

def overlay_cam(img: Image.Image, cam: np.ndarray, alpha_img=0.55, alpha_heat=0.45):
    """Overlay CAM on image using PIL+numpy (no cv2)."""
    img_np = np.array(img).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]

    cam_u8 = (np.clip(cam, 0, 1) * 255).astype(np.uint8)
    cam_resized = Image.fromarray(cam_u8).resize((w, h), resample=Image.BILINEAR)
    cam_resized = np.array(cam_resized).astype(np.float32) / 255.0

    heat = jet_colormap(cam_resized)  # (h,w,3) float 0..1

    out = alpha_img * img_np + alpha_heat * heat
    out = np.clip(out, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("### ตั้งค่า")
    th = st.slider("เกณฑ์ไม่แน่ใจ", 0.40, 0.90, THRESHOLD_DEFAULT, 0.01)
    img_size = st.select_slider("ขนาดภาพ", [320, 384, 448], IMG_SIZE_DEFAULT)
    st.caption(f"Device: {DEVICE}")

# -------------------- Header (fixed: no white hard-to-read box) --------------------
st.markdown("""
<div class="card">
  <div class="title">FoodVision Guard — Bread</div>
  <div class="sub">คัดกรองขนมปังที่มีเชื้อราด้วย Deep Learning (EfficientNet-B0 + Grad-CAM)</div>
</div>
""", unsafe_allow_html=True)

# -------------------- Load model --------------------
model, cam_engine, tfm = load_model(img_size)

# -------------------- Upload --------------------
uploaded = st.file_uploader("อัปโหลดภาพขนมปัง", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
    latency = (time.time() - t0) * 1000

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    idx = int(np.argmax(probs))
    pred = CLASSES[idx]
    conf = float(probs[idx])

    if conf < th:
        label = "ไม่แน่ใจ"
        advice = "แนะนำหลีกเลี่ยงการบริโภค"
    else:
        label = "มีรา" if pred == "mold" else "สะอาด"
        advice = "ควรทิ้ง" if pred == "mold" else "ยังสามารถบริโภคได้"

    cam = cam_engine(x, idx)
    cam_img = overlay_cam(img, cam)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="ภาพต้นฉบับ", use_container_width=True)
        st.image(cam_img, caption="Grad-CAM", use_container_width=True)

    with col2:
        st.markdown(f"### ผลลัพธ์: **{label}**")
        st.metric("ความมั่นใจ", f"{conf:.2f}")
        st.metric("เวลา", f"{latency:.0f} ms")
        st.info(advice)

        df = pd.DataFrame({"Class": CLASSES, "Probability": probs})
        st.bar_chart(df.set_index("Class"))

st.caption("⚠️ ใช้เพื่อคัดกรองเบื้องต้น ไม่ทดแทนการตรวจในห้องปฏิบัติการ")
