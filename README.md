##  Understanding U-Net and Multi-level U-Net

###  What is U-Net?

U-Net is a **variant of Autoencoder** designed for **image segmentation** tasks.  
It gets its name from its **U-shaped architecture**, consisting of:

- **Encoder (Contracting Path):** Downsampling layers that extract semantic features, but reduce spatial resolution.  
- **Decoder (Expanding Path):** Upsampling layers that reconstruct the segmentation mask back to input size.  
- **Skip Connections:** Direct links between encoder and decoder layers (see gray arrows in Fig.1) to preserve fine-grained details that would otherwise be lost.  

> Without skip connections (like in a vanilla autoencoder), small details such as lesions or tumors might be filtered out during feature compression.

<p align="center">
  <img src="https://ithelp.ithome.com.tw/upload/images/20200919/20001976muQqJZ6VAC.png" width="600" />
</p>
<p align="center">Fig.1. U-Net architecture (source: Deep Learning for Image Segmentation: U-Net)</p>
*Source:* [Deep Learning for Image Segmentation: U-Net Architecture](https://fritz.ai/deep-learning-for-image-segmentation-u-net-architecture/)

---

###  Why U-Net Works for Medical Imaging

Medical image segmentation requires **pixel-level precision**, often detecting **tiny abnormal regions** (e.g., tumors).  
Skip connections allow U-Net to combine **low-level spatial details** (edges, textures) with **high-level semantic context**, making it highly effective for biomedical tasks.

<p align="center">
<img width="279" height="300" alt="image" src="https://github.com/user-attachments/assets/e1cfb8c8-0563-4efc-83e0-445afadfacec" />
<img width="279" height="300" alt="image" src="https://github.com/user-attachments/assets/7d3a59ea-d257-4aa6-a352-fb5b725c5e0b" />
<img width="279" height="300" alt="image" src="https://github.com/user-attachments/assets/1e84cb0b-bb32-44df-9827-87fa76283276" />
</p>
<p align="center">Fig.2. Example: Original CT slice, ground-truth mask, U-Net segmentation result</p>
*Source:* [Day 20：使用 U-Net 作影像分割(Image Segmentation)](https://ithelp.ithome.com.tw/articles/10240314)

---

### 🔹 From U-Net to Multi-level U-Net

While a single U-Net works reasonably well for **organ-level segmentation**, it struggles when applied directly to **tumors**:
- Tumors are **small, highly imbalanced**, and often blend with surrounding tissues.  
- The **liver itself is anatomically divided into multiple lobes/segments**, making boundary delineation even harder.  
<img width="500" height="282" alt="image" src="https://github.com/user-attachments/assets/c6bb8864-7039-4281-9347-30b1cad3479f" />
*Source:* [Wikipedia](https://zh.wikipedia.org/zh-tw/%E8%82%9D%E8%87%9F)

- A single-pass U-Net tends to miss fine structures or merge them into larger regions.  

---

###  My Multi-level U-Net Design

1. **Stage 1 — Liver ROI Extraction**  
   - A first-pass U-Net isolates the **entire liver region** from CT volumes.  
   - Provides a coarse mask to crop out irrelevant organs and reduce search space.  

2. **Stage 2 — Multi-level U-Net (Liver Segments + Tumors)**  
   - Within the extracted liver ROI, a **Multi-level U-Net** performs:  
     - **Liver segmentation by anatomical segments** (lobes/regions).  
     - **Simultaneous tumor segmentation** inside each segment.  
   - This design ensures tumors are detected **in context of their liver region**, improving sensitivity and clinical interpretability.  

3. **Post-processing & Reporting**  
   - Apply 3D connected components and morphological filtering.  
   - Extract lesion-level statistics (size, count, segment location).  
   - Generate a **structured report** (tumor distribution per liver segment).
---

###  Benefits
- **Higher tumor sensitivity:** More small lesions detected.  
- **Better boundary accuracy:** Liver segmentation isolates anatomical regions before tumor detection.  
- **Clinical alignment:** Outputs structured reports (tumor size, count, lobe/segment location).  

---

<p align="center">
  <img src="./assets/multi_level_unet.png" width="650" />
</p>
<p align="center"><b>Fig.3.</b> Multi-level U-Net pipeline: Liver segmentation → Tumor segmentation → Structured report</p>

##  Why Multi-level U-Net?

Unlike a single U-Net that segments all structures in **one pass**, the **Multi-level U-Net** pipeline performs segmentation in **two stages**:

1. **Stage 1 — Organ-level segmentation** (e.g., liver, surrounding tissues).  
2. **Stage 2 — Fine segmentation inside the organ ROI** (e.g., tumor regions inside the liver).  

This decomposition significantly improves accuracy for **small, low-contrast, or overlapping regions** that a single U-Net would often miss.

<p align="center">
  <img src="./assets/multi_level_unet.png" width="650" />
</p>
<p align="center">Fig.4. Multi-level U-Net segmentation results across multiple organs and regions.</p>

---

###  Challenges
- Requires **two models** (liver U-Net + tumor U-Net) to run sequentially.  
- Training is more complex — error propagation from Stage 1 affects Stage 2.  
- More **compute time & memory** compared to single U-Net.  

---

###  Advantages
- **Higher accuracy:** Captures fine tumor boundaries (Dice ↑ by ~5–10%).  
- **Better separation:** Multi-organ context improves region isolation.  
- **Clinical utility:** Structured outputs (size, count, stage hints) align better with diagnostic workflows.  
- **Robustness:** Handles small lesions that are usually ignored by single U-Net.  

In practice: Though harder to train and slower to run, the **Multi-level U-Net** achieves significantly **better precision and clinical relevance**.

##  Workflow (Step-by-Step)

1. **Preprocessing**  
   - Load CT volumes (DICOM / NIfTI).  
   - Apply **HU windowing** (liver window), **resampling** to isotropic spacing, and **z-score normalization**.  
   - (Optional) Artifact removal and abdominal cropping.  

2. **Stage 1 — Liver Segmentation (U-Net, Level 1)**  
   - Input: Preprocessed CT slices.  
   - Output: Binary **liver mask**.  
   - Performance: Dice score **> 0.88**.  

3. **Stage 2 — Tumor Segmentation (U-Net, Level 2)**  
   - Input: CT restricted to **liver ROI** (from Stage 1).  
   - Output: Binary **tumor mask** within liver region.  
   - Performance: Dice score **> 0.81**.  

Together, these form a **Multi-level U-Net pipeline**, where **Stage 1 guides Stage 2**, improving tumor detection precision compared to single-stage U-Net.

4. **Post-processing**  
   - Apply 3D Connected Component Labeling.  
   - Morphological ops (closing/opening, small-blob removal).  
   - Compute lesion-level statistics (volume, diameter, centroid).  

5. **Structured Report Generation**  
   - Summarize tumor **count, size, location, and stage hints**.  
   - Export **JSON/PDF reports** with images and metadata.  
   - Radiologist can **review/edit/approve** via Web UI.  

6. **Clinical Review & Database**  
   - Reports and segmentation masks stored in **MySQL/MongoDB**.  
   - Web UI (PHP) enables easy browsing, editing, and exporting to PACS/RIS.  

---

##  Model Design

- **Architecture:** **Multi-level U-Net**  
  - **Level 1:** Segments whole liver (coarse mask).  
  - **Level 2:** Segments tumors only inside the liver mask (refined ROI).  
- **Why Multi-level?**  
  - Traditional single U-Net struggles with small tumors & class imbalance.  
  - By isolating liver ROI first, Level 2 avoids false positives outside the liver.  
  - Improves **boundary detail** and **tumor sensitivity**.  
- **Input Strategy:** 2.5D slice stacking or 3D patches.  
- **Loss Functions:**  
  - Dice Loss (overlap accuracy).  
  - Focal Loss (handle imbalance between tumor vs background).  
- **Augmentation:** Random rotation, flip, intensity jitter, elastic deformation.  
- **Optimization:** Adam optimizer, learning rate decay, AMP mixed precision.  

---

##  Dataset & Metrics

- **Dataset:** De-identified CT scans provided by **Kaohsiung Chang Gung Memorial Hospital (CGMH)** under IRB approval.  
- **Splits:** Train 70% · Validation 10% · Test 20%  
- **Metrics:** Dice, Hausdorff Distance, Average Surface Distance.  

| Task         | Model             | Metric | Result (example) |
|--------------|------------------|--------|------------------|
| Liver Mask   | U-Net (Level 1)  | Dice   | **0.88 – 0.91**  |
| Tumor Mask   | U-Net (Level 2)  | Dice   | **0.81 – 0.84**  |
| Report Gen.  | Multi-level Flow | Time saved | **-60% / case** vs manual review |

---

##  Repository Layout

```text
.
├─ src/
│  ├─ preprocess/           # CT preprocessing (windowing, resample, normalize)
│  ├─ models/               # unet_liver.py (Stage 1), unet_tumor.py (Stage 2)
│  ├─ train/                # dataloaders, losses, trainers
│  ├─ infer/                # multi-level inference pipeline
│  ├─ postprocess/          # cc3d, morphology, lesion statistics
│  ├─ report/               # json_builder.py, pdf_export.py
│  └─ webui/                # React JS web interface for report review/export
├─ configs/
│  ├─ liver.yaml            # configs for Stage 1 (liver)
│  └─ tumor.yaml            # configs for Stage 2 (tumor)
├─ data/                    # (gitignored) sample CT scans
└─ README.md
```
