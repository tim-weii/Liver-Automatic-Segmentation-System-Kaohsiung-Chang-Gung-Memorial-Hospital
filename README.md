
#  Liver Automatic Segmentation System

**Institution:** Kaohsiung Chang Gung Memorial Hospital (2024/05) — 🥇 *Best Project Award (1st place)*  
**Team:** 4 persons · **My Contribution:** 50%

> A **Multi-level U-Net system** that segments liver and tumors from CT scans and generates **structured clinical reports**, reducing radiologists’ workload, shortening diagnosis time, improving accuracy, and accelerating clinical decisions.

---

## 📖 Understanding U-Net and Multi-level U-Net

### 🔹 What is U-Net?

U-Net is a **variant of Autoencoder** designed for **image segmentation** tasks.  
It gets its name from its **U-shaped architecture**, consisting of:

- **Encoder (Contracting Path):** Downsampling layers that extract semantic features while reducing spatial resolution.  
- **Decoder (Expanding Path):** Upsampling layers that reconstruct segmentation masks back to input size.  
- **Skip Connections:** Direct links between encoder and decoder layers preserve fine-grained spatial details.  

Without skip connections, small details such as **tumors** would often be lost during feature compression.

<p align="center">
  <img src="https://ithelp.ithome.com.tw/upload/images/20200919/20001976muQqJZ6VAC.png" width="600" />
</p>
<p align="center">Fig.1. U-Net architecture (source: Deep Learning for Image Segmentation)</p>

---

### 🔹 Why U-Net Works for Medical Imaging

Medical image segmentation requires **pixel-level precision**, often to detect **tiny abnormalities** (e.g., tumors).  
Skip connections allow U-Net to combine **low-level spatial details** (edges, textures) with **high-level semantic context**, making it highly effective for biomedical tasks.

<p align="center">
<img width="279" height="300" alt="image" src="https://github.com/user-attachments/assets/e1cfb8c8-0563-4efc-83e0-445afadfacec" />
<img width="279" height="300" alt="image" src="https://github.com/user-attachments/assets/7d3a59ea-d257-4aa6-a352-fb5b725c5e0b" />
<img width="279" height="300" alt="image" src="https://github.com/user-attachments/assets/1e84cb0b-bb32-44df-9827-87fa76283276" />
</p>
<p align="center">Fig.2. Example: Original CT slice, ground-truth mask, U-Net segmentation result</p>  

*Source:* [Day 20：使用 U-Net 作影像分割](https://ithelp.ithome.com.tw/articles/10240314)

---

##  Why Multi-level U-Net?

Traditional U-Net has shown strong performance in medical image segmentation, but when applied directly to **liver CT scans with tumors**, it encounters several fundamental challenges.  
To overcome these, we designed a **Multi-level U-Net**, where Stage 1 isolates the liver region and Stage 2 performs **segment-level + tumor segmentation** inside that ROI.  

The motivation comes from three critical issues:

---

1. **Severe Class Imbalance**  
   - Tumors typically occupy **<2% of total voxels**, while the liver and background dominate.  
   - A single U-Net tends to optimize for the **global Dice score**, prioritizing large regions and causing **tiny tumors to vanish during downsampling**.  
   - We explicitly handle this imbalance (ROI cropping + loss design such as **Focal/Dice Loss**) to preserve small lesions.  

---

2. **Anatomical Complexity of the Liver**  
   - The liver is divided into **8 Couinaud segments**, each with distinct vascular structures and clinical significance.  
   - A single-pass segmentation often **blurs inter-segment boundaries**, making it difficult to localize tumors relative to anatomy.  
   - Clinically, a mask that only says *“tumor detected”* is insufficient. Radiologists need structured outputs like:  
     *“tumor in Segment IVa, diameter 2.3 cm.”*  
   - Multi-level segmentation enforces **segment-aware tumor mapping**, bridging model outputs with real clinical reporting.  

---

3. **Context Dilution & Efficiency**  
   - Feeding the **entire abdomen** into one network wastes capacity on irrelevant organs (kidneys, stomach, vessels).  
   - This leads to **higher false positives**, slower convergence, and reduced sensitivity to subtle patterns like vessels or small lesions.  
   - By cropping to the liver ROI first, the Multi-level U-Net:  
     - Allocates computation **only to the liver and tumors**.  
     - Improves accuracy on **fine-grained variations** (boundaries, vascular adjacency).  
     - Increases training efficiency and reduces unnecessary noise.  

---

##  Our System Design

1. **Stage 1 — Liver ROI Extraction**  
   - First-pass U-Net isolates the **entire liver region** from CT scans.  
   - Provides a **coarse mask** to crop irrelevant organs and reduce search space.  

2. **Stage 2 — Multi-level U-Net (Liver Segments + Tumors)**  
   - Input restricted to the **liver ROI from Stage 1**.  
   - Performs **two tasks simultaneously**:  
     - Segmenting liver by **anatomical segments** (lobes/regions).  
     - Segmenting **tumors** within each region.  
   - Tumors are detected **in anatomical context**, improving sensitivity and interpretability.  

3. **Post-processing & Reporting**  
   - Apply 3D connected components, morphological filtering.  
   - Extract **lesion statistics** (size, count, location).  
   - Generate a **structured report** (tumor distribution by liver segment).  
---

##  Benefits of Multi-level U-Net (vs Single U-Net)

| Aspect | Single U-Net (Baseline) | Multi-level U-Net (Our Design) | Benefit |
|--------|--------------------------|--------------------------------|---------|
| **Small Tumors** | Vanish due to class imbalance (<2% voxels) | ROI cropping + focal loss enhance tumor-to-background ratio | ✅ Better small tumor detection |
| **Boundary Accuracy** | Blurs across liver lobes, hard to localize | Segments **by anatomical regions**, preserving fine boundaries | ✅ More precise tumor boundaries |
| **Clinical Interpretability** | Only says “tumor present” | Reports **tumor size + segment location (e.g., IVa, 2.3 cm)** | ✅ Directly usable in diagnosis |
| **False Positives** | Wasted capacity on irrelevant organs → more FP | Ignores kidneys/stomach/etc., focuses on **liver-only ROI** | ✅ Fewer false positives |
| **Supervision Signal** | Binary tumor mask only | Multi-task: **segments + tumors** → richer gradients | ✅ Stronger learning signal |
| **Efficiency & Workflow** | Radiologists must manually verify every CT slice | Structured reports summarize count, size, region → quick review | ✅ Saves time, reduces fatigue/misdiagnosis |

---

👉 **Summary:**  
By constraining the model to liver ROI and combining **multi-task supervision (liver + tumors)**, the Multi-level U-Net not only boosts **accuracy** but also aligns outputs with **clinical workflows**, helping radiologists save time and make safer decisions.



## 🔄 Workflow (Step-by-Step)

1. **Preprocessing**  
   - Load CT volumes (DICOM/NIfTI).  
   - Apply **HU windowing**, resample to isotropic spacing, normalize.  

2. **Stage 1 — Liver Segmentation (U-Net, Level 1)**  
   - Input: preprocessed CT slices.  
   - Output: binary **liver mask**.  
   - Performance: Dice **>0.88**.  

3. **Stage 2 — Tumor Segmentation (U-Net, Level 2)**  
   - Input: liver ROI only.  
   - Output: binary **tumor mask**.  
   - Performance: Dice **>0.81**.  

4. **Post-processing**  
   - 3D connected components + morphology.  
   - Compute lesion statistics: **volume, diameter, centroid**.  

5. **Structured Report Generation**  
   - Summarize tumor **count, size, location, stage hints**.  
   - Export **JSON/PDF**. Radiologists can review/edit in Web UI.  

6. **Clinical Review & Database**  
   - Reports + masks stored in **MySQL**.  
   - **Web UI (ReactJS)** supports browsing, editing, export.  

---

## 🧠 Model Design

- **Architecture:** Multi-level U-Net  
  - Stage 1: whole-liver segmentation  
  - Stage 2: tumor segmentation inside liver ROI  
- **Loss Functions:** Dice Loss + Focal Loss (handles imbalance).  
- **Input Strategy:** 2.5D slice stacking or 3D patches.  
- **Optimization:** Adam + learning rate decay, AMP mixed precision.  
- **Augmentation:** rotation, flip, intensity jitter, elastic deformation.  
- **Advantage:** Reduces false positives outside liver; improves tumor boundary precision.  

---

## 📊 Dataset & Metrics

- **Dataset:** De-identified CT scans from **Kaohsiung Chang Gung Memorial Hospital**, IRB-approved.  
- **Splits:** Train 70% · Validation 10% · Test 20%  
- **Metrics:** Dice, Hausdorff Distance, Average Surface Distance  

| Task         | Model            | Metric | Result |
|--------------|-----------------|--------|--------|
| Liver Mask   | U-Net (Stage 1) | Dice   | **0.88–0.91** |
| Tumor Mask   | U-Net (Stage 2) | Dice   | **0.81–0.84** |
| Report Gen.  | Multi-level Flow | Time saved | **-60% / case** |

---

## ⚖️ Challenges

- **Imbalance:** Tumors occupy **<2% of voxels**; naïve models ignore them.  
- **3D vs 2.5D:** 3D context improves results but is memory-intensive; we use hybrid 2.5D.  
- **Annotation noise:** Radiologists vary; post-processing reduces false lesions.  
- **Two-stage complexity:** Errors from Stage 1 propagate into Stage 2.  

---

## 📈 Clinical Impact

- Radiologists review **AI-assisted structured reports** instead of manual slice-by-slice labeling.  
- **~60% time saved per case** in clinical workflow.  
- Improves consistency, reduces fatigue-related misdiagnosis risk.  

---

## 🔮 Future Work

- **Advanced models:** Attention U-Net, TransUNet, Swin-UNet.  
- **Semi-supervised learning:** leverage unlabeled CTs.  
- **Multi-modal fusion:** integrate MRI + CT.  
- **Deployment:** export structured results as **DICOM-SR** for direct PACS integration.  

---

## 📂 Repository Layout

```text
.
├─ src/
│  ├─ preprocess/           # CT preprocessing
│  ├─ models/               # unet_liver.py, unet_tumor.py
│  ├─ train/                # dataloaders, losses, trainers
│  ├─ infer/                # multi-level inference pipeline
│  ├─ postprocess/          # cc3d, morphology, lesion statistics
│  ├─ report/               # json_builder.py, pdf_export.py
│  └─ webui/                # PHP web interface for review/export
├─ configs/
│  ├─ liver.yaml            # configs for Stage 1 (liver)
│  └─ tumor.yaml            # configs for Stage 2 (tumor)
├─ data/                    # (gitignored) sample CT scans
└─ README.md
```
