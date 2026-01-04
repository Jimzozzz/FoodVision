import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import time
import cv2
import io
import pandas as pd
from datetime import datetime

# -------------------- Page config --------------------
st.set_page_config(page_title="FoodVision Guard — Bread", layout="wide")

# -------------------- CSS --------------------
st.markdown("""
<style>
html, body, [class*="css"] {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Noto Sans Thai', sans-serif;
  color: #0f172a;
}
:root { --bg:#ffffff; --muted:#f1f5f9; --border:#e2e8f0; --sub:#475569; }
.block-container { padding-top: 16px; padding-bottom: 24px; }
.card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}
.metric { border:1px solid var(--border); border-radius:10px; padding:12px; background:var(--muted); }
#MainMenu, footer, header { visibility: hidden; }

/* ✅ ทำให้ “กรอบหัวข้อ” พื้นขาว + ตัวหนังสือสีดำ (override ชัวร์) */
.card.light{
  background: #ffffff;
  border: 1px solid #e2e8f0;
}
.card.light, .card.light *{
  color: #0f172a !important;
}
.card.light .sub{
  color: #334155 !important;
}
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

def overlay_cam(img, cam):
    img_np = np.array(img)
    cam = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_np, 0.55, heatmap, 0.45, 0)
    return Image.fromarray(out)

# -------------------- Sidebar --------------------
with st.sidebar:
    th = st.slider("เกณฑ์ไม่แน่ใจ", 0.40, 0.90, THRESHOLD_DEFAULT, 0.01)
    img_size = st.select_slider("ขนาดภาพ", [320, 384, 448], IMG_SIZE_DEFAULT)
    st.caption(f"Device: {DEVICE}")

# -------------------- Header --------------------
# ✅ เปลี่ยนให้กรอบหัวข้อเป็นพื้นขาว + ตัวหนังสือดำ
st.markdown("""
<div class="card light">
<b>Bread Mold Detection System Using Deep Learning</b><br>
<span class="sub">ระบบตรวจจับเชื้อราในขนมปังด้วย Deep Learning</span>
</div>
""", unsafe_allow_html=True)

# -------------------- Load model --------------------
model, cam_engine, tfm = load_model(img_size)

# -------------------- Upload --------------------
uploaded = st.file_uploader("อัปโหลดภาพขนมปัง", type=["jpg","png","jpeg"])

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
    conf = probs[idx]

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

st.caption("⚠️ ใช้เพื่อคัดกรองเบื้องต้น ไม่สามารถทดแทนผลการตรวจจากห้องปฏิบัติการ")
