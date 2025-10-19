# FoodVision Guard – AI คัดกรองอาหารจากภาพ
ครอบคลุม 3 สโคป: (1) ราในขนมปัง/เบเกอรี่ (2) ผลไม้เริ่มช้ำ/เสีย (เลือกชนิดเดียว) (3) ความสดของปลา (ตา/เหงือก/ผิว)

## โครงสร้างโฟลเดอร์
```
FoodVisionGuard/
├── configs/
│   ├── bread_mold.yaml
│   ├── fruit_bruise.yaml
│   └── fish_freshness.yaml
├── data/               # วางรูปตามที่กำหนดใน config
├── foodvision_guard/
│   ├── datasets.py
│   ├── models.py
│   ├── train.py
│   ├── eval.py
│   ├── gradcam.py
│   └── utils.py
├── scripts/
│   ├── train_bread.sh
│   ├── train_fruit.sh
│   └── train_fish.sh
├── requirements.txt
└── README.md
```

## วิธีเริ่มเร็วที่สุด
1) สร้างชุดภาพตาม config (ดูตัวอย่างใน `configs/*.yaml`)
2) ติดตั้งไลบรารี: `pip install -r requirements.txt`
3) เทรน: `bash scripts/train_bread.sh` (หรือ fruit/fish)
4) ดู heatmap: รัน `python -m foodvision_guard.gradcam --config configs/bread_mold.yaml --img <path_to_image>`

## เกณฑ์วัดผล (ตั้งเป้า)
- Bread/Fruit: Accuracy ≥ 90%, F1-macro ≥ 0.88 (test ข้ามวัน/ล็อต)
- Fish (head ละ): F1 ≥ 0.88; FreshScore AUC ≥ 0.90
- Latency (Edge): classifier ≤ 150 ms/ภาพ

## หมายเหตุความปลอดภัย
- ระบบนี้เป็น **เครื่องมือคัดกรองเบื้องต้น** ไม่ใช่ผลทดแทนแล็บ/ผู้เชี่ยวชาญ
- ปลา: ไม่รองรับกรณีเคยแช่แข็ง-ละลาย
- ขนมปัง: พบรา ⇒ แนะนำทิ้งทั้งก้อน
