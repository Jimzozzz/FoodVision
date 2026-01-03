# app_streamlit.py — FoodVision Guard (Bread) · Minimal UI (no Entropy card)

import streamlit as st
import torch, torch.nn as nn
from torchvision import models, transforms, datasets
from PIL import Image
import numpy as np, time, cv2, io, pandas as pd
from datetime import datetime

# ---------- Page Config ----------
st.set_page_config(page_title="FoodVision Guard — Bread", layout="wide")

# ---------- Global Styles ----------
st.markdown("""
<style>
html, body, [class*="css"] { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, 'Noto Sans Thai', sans-serif; color:#0f172a; }
:root { --bg:#ffffff; --muted:#f1f5f9; --border:#e2e8f0; --sub:#475569; }
.block-container { padding-top:16px; padding-bottom:24px; }
.min-topbar { display:flex; align-items:center; justify-content:space-between; padding:14px 18px; background:var(--bg); border:1px solid var(--border); border-radius:12px; margin-bottom:14px; }
.min-title { font-size:20px; font-weight:600; }
.min-sub { color:var(--sub); font-size:14px; }
.card { background:var(--bg); border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
.metric-row { display:grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap:10px; } /* 3 กล่อง: ผลลัพธ์, ความมั่นใจ, เวลา */
.metric { border:1px solid var(--border); border-radius:10px; padding:12px 12px; background:var(--muted); }
.metric .k { font-size:20px; font-weight:700; }
.metric .l { font-size:12px; color:var(--sub); }
#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ---------- Constants ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = "best_bread_mold.pt"
TRAIN_DIR = "data/bread/train"
IMG_SIZE_DEFAULT = 384
THRESHOLD_DEFAULT = 0.60

# ---------- Grad-CAM ----------
class GradCAM:
    def __init__(self, model, target_layer: nn.Module):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        def fwd_hook(m, i, o): self.activations = o.detach().clone()
        def bwd_hook(m, gi, go): self.gradients = go[0].detach().clone()
        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, x, class_idx=None):
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(1).item())
        self.model.zero_grad(set_to_none=True)
        logits[:, class_idx].sum().backward()
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
        return cam[0, 0].cpu().numpy(), class_idx

def disable_inplace_activations(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.SiLU, nn.ReLU)):
            m.inplace = False

@st.cache_resource
def load_model_and_tfm(weights_path: str, train_dir: str, img_size: int):
    ds = datasets.ImageFolder(train_dir); classes = ds.classes
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    except Exception:
        model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(classes))
    disable_inplace_activations(model)
    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)

    target_layer = model.features[-1][0]  # Conv2d
    cam_engine = GradCAM(model, target_layer)

    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return model, classes, cam_engine, tfm

def preprocess(pil_img: Image.Image, tfm): return tfm(pil_img).unsqueeze(0).to(DEVICE)

# --- Probabilities helpers ---
def softmax(logits):  # raw softmax
    return torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

def softmax_with_temp(logits, T: float = 1.0):  # temperature scaling
    z = logits / max(T, 1e-6)
    return torch.softmax(z, dim=1)[0].detach().cpu().numpy()

def entropy_norm(p: np.ndarray):
    eps = 1e-12
    h = -np.sum(p * np.log(p + eps))
    h_max = np.log(len(p))
    return float(h / (h_max + eps))

def overlay_cam(pil_img: Image.Image, cam: np.ndarray) -> Image.Image:
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)

def probs_df(classes, probs):
    return pd.DataFrame({"class": classes, "probability": probs}).sort_values("probability", ascending=False)

def confidence_level(p_max: float, ent: float):
    if p_max >= 0.90 and ent <= 0.15:
        return "มั่นใจมาก"
    elif p_max >= 0.70 and ent <= 0.35:
        return "มั่นใจปานกลาง"
    else:
        return "ไม่แน่ใจ"

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("#### การตั้งค่า")
    th = st.slider("เกณฑ์ความเชื่อมั่นขั้นต่ำ (Confidence Threshold)", 0.40, 0.90, THRESHOLD_DEFAULT, 0.01)
    img_size = IMG_SIZE_DEFAULT  # ล็อกที่ 384
    T = st.slider("Temperature (T)", 0.5, 3.0, 1.0, 0.1, help="T>1 ลดความพุ่งของค่า softmax; T<1 ทำให้พุ่งขึ้น")
    st.markdown("---")
    st.caption(f"อุปกรณ์: {DEVICE}")
    st.caption(f"น้ำหนักโมเดล: {WEIGHTS_PATH}")

# ---------- Topbar ----------
st.markdown("""
<div class="min-topbar">
  <div class="min-title">FoodVision Guard — Bread</div>
  <div class="min-sub">คัดกรองภาพขนมปัง: สะอาด / มีรา พร้อม Grad-CAM และค่าความไม่แน่นอน</div>
</div>
""", unsafe_allow_html=True)

# load model
model, classes, cam_engine, tfm = load_model_and_tfm(WEIGHTS_PATH, TRAIN_DIR, img_size)
if "history" not in st.session_state: st.session_state.history = []

# ---------- Upload ----------
left, right = st.columns([3,2], gap="large")
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("อัปโหลดภาพ")
    multi = st.toggle("โหมดหลายภาพ", value=False)
    if multi:
        uploaded = st.file_uploader("เลือกรูป .jpg/.png", type=["jpg","jpeg","png"], accept_multiple_files=True)
    else:
        uploaded = st.file_uploader("เลือกรูป .jpg/.png", type=["jpg","jpeg","png"], accept_multiple_files=False)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("แนวทางการถ่ายภาพ")
    st.markdown("- ใช้แสงสีขาวและพื้นหลังเรียบ\n- เห็นเนื้อขนมปังชัดเจน\n- เลี่ยงเงาและแสงส้มจัด")
    st.markdown('</div>', unsafe_allow_html=True)

def predict_one(file):
    img = Image.open(file).convert("RGB")
    x = preprocess(img, tfm)

    # inference
    t0 = time.time()
    with torch.no_grad():
        logits = model(x)
    latency_ms = (time.time() - t0) * 1000.0

    # probs (raw & calibrated)
    probs_raw = softmax(logits)
    probs_cal = softmax_with_temp(logits, T=T)

    pred_idx = int(np.argmax(probs_cal))
    pred_name = classes[pred_idx]
    p_max_raw = float(np.max(probs_raw))
    p_max_cal = float(np.max(probs_cal))

    # uncertainty (ใช้ภายใน ไม่โชว์การ์ด)
    ent = entropy_norm(probs_cal)
    level = confidence_level(p_max_cal, ent)

    # decision
    if p_max_cal < th:
        label = "ไม่แน่ใจ"
        advice = "แนะนำหลีกเลี่ยงการบริโภค และตรวจซ้ำด้วยสายตาหรือกลิ่น"
    else:
        label = "มีรา" if pred_name == "moldy" else "สะอาด"
        advice = "พบสัญญาณรา: แนะนำทิ้งทั้งก้อน" if pred_name == "moldy" else "ยังไม่พบสัญญาณรา: โปรดตรวจวันหมดอายุและกลิ่นประกอบ"

    # Grad-CAM
    cam, _ = cam_engine(x, class_idx=pred_idx)
    img_cam = overlay_cam(img, cam)

    # thumbnail
    thumb = img.copy(); thumb.thumbnail((240, 240))
    buf = io.BytesIO(); thumb.save(buf, format="JPEG", quality=85)

    return {
        "image": img, "img_cam": img_cam, "classes": classes,
        "probs_raw": probs_raw, "probs_cal": probs_cal,
        "p_max_raw": p_max_raw, "p_max_cal": p_max_cal,
        "entropy": ent, "level": level,
        "label": label, "pred_name": pred_name,
        "latency_ms": latency_ms,
        "advice": advice, "thumb": buf.getvalue(),
        "filename": getattr(file, "name", "uploaded.jpg")
    }

# ---------- Inference & Result ----------
if uploaded:
    files = uploaded if isinstance(uploaded, list) else [uploaded]
    for file in files:
        out = predict_one(file)

        st.session_state.history.insert(0, {
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": out["filename"],
            "label": out["label"],
            "prob": out["p_max_cal"],
            "lat": out["latency_ms"],
            "thumb": out["thumb"],
        })

        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns([2,3], gap="large")

        with c1:
            st.image(out["image"], caption="ภาพที่อัปโหลด", use_container_width=True)
            with st.expander("จุดที่เข้าข่าย (Grad-CAM)"):
                st.image(out["img_cam"], caption="Grad-CAM", use_container_width=True)

        with c2:
            st.subheader("ผลการประเมิน")
            st.markdown('<div class="metric-row">', unsafe_allow_html=True)
            # 3 metrics: ผลลัพธ์ / ความมั่นใจ / เวลา
            m1, m2, m3 = st.columns(3)
            m1.markdown(f"<div class='metric'><div class='l'>ผลลัพธ์</div><div class='k'>{out['label']}</div></div>", unsafe_allow_html=True)
            m2.markdown(f"<div class='metric'><div class='l'>ความมั่นใจ </div><div class='k'>{out['p_max_cal']:.2f}</div></div>", unsafe_allow_html=True)
            m3.markdown(f"<div class='metric'><div class='l'>เวลา</div><div class='k'>{out['latency_ms']:.0f} ms</div></div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # แสดง bar chart ความน่าจะเป็นรายคลาส (หลังปรับ T)
            df = probs_df(out["classes"], out["probs_cal"])
            st.bar_chart(df.set_index("class"))

            st.info(out["advice"])
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- History ----------
st.markdown("#### ประวัติรูป")
if len(st.session_state.history) == 0:
    st.caption("อัปโหลดรูปเพื่อเริ่มบันทึกประวัติ")
else:
    cols = st.columns(5)
    for i, h in enumerate(st.session_state.history[:10]):
        with cols[i % 5]:
            st.image(h["thumb"], caption=f"{h['label']} ({h['prob']:.2f})", use_container_width=True)

st.caption("ข้อจำกัด: เครื่องมือนี้ใช้คัดกรองเบื้องต้น ไม่ทดแทนการตรวจวิเคราะห์ในห้องปฏิบัติการ")
