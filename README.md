
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

### 🔹 From U-Net to Multi-level U-Net

While a single U-Net works for **organ-level segmentation**, it struggles with **tumors**:
- Tumors are **small, imbalanced**, and often blend with surrounding tissue.  
- The **liver is divided into anatomical segments** (lobes/regions), making boundary delineation harder.  
- A single-pass model tends to miss fine structures or merge them incorrectly.  

---

## 🔄 Our Multi-level U-Net Design

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

### ✅ Benefits

- **Higher tumor sensitivity:** More small lesions detected.  
- **Better boundary accuracy:** Anatomical segmentation improves tumor delineation.  
- **Clinical alignment:** Structured reports (size, count, lobe/segment location).  

<p align="center">
  <img src="./assets/multi_level_unet.png" width="650" />
</p>
<p align="center"><b>Fig.3.</b> Multi-level U-Net pipeline: Liver segmentation → Tumor segmentation → Structured report</p>

---

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
   - Reports + masks stored in **MySQL/MongoDB**.  
   - **Web UI (PHP)** supports browsing, editing, PACS/RIS export.  

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
