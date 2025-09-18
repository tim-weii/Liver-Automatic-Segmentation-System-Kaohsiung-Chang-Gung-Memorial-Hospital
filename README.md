## 🧬 Understanding U-Net and Multi-level U-Net

### 🔹 What is U-Net?

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

---

### 🔹 Why U-Net Works for Medical Imaging

Medical image segmentation requires **pixel-level precision**, often detecting **tiny abnormal regions** (e.g., tumors).  
Skip connections allow U-Net to combine **low-level spatial details** (edges, textures) with **high-level semantic context**, making it highly effective for biomedical tasks.

<p align="center">
  <img src="https://ithelp.ithome.com.tw/upload/images/20200920/20001976RwTZHYSSbr.png" width="250" />
  <img src="https://ithelp.ithome.com.tw/upload/images/20200920/20001976W7GE2cMYsS.png" width="250" />
  <img src="https://ithelp.ithome.com.tw/upload/images/20200920/20001976oTl8JrUc8N.png" width="250" />
</p>
<p align="center">Fig.2. Example: Original CT slice, ground-truth mask, U-Net segmentation result</p>

---

### 🔹 From U-Net to Multi-level U-Net

While a single U-Net works well for organs, **tumors are small, imbalanced, and hard to detect**.  
Thus, we design a **Multi-level U-Net**:

1. **Stage 1 (Liver U-Net):** Coarse segmentation of the **entire liver region**.  
2. **Stage 2 (Tumor U-Net):** Input restricted to **liver ROI** from Stage 1, focusing only on tumors.  
3. **Post-processing:** 3D connected components, morphological filtering, lesion statistics.  

This two-stage pipeline significantly improves **tumor sensitivity and boundary accuracy** compared to a single U-Net.

<p align="center">
  <img src="https://github.com/user-attachments/assets/your-custom-multiunet-diagram.png" width="600" />
</p>
<p align="center">Fig.3. Multi-level U-Net pipeline: Liver segmentation → Tumor segmentation → Structured report</p>

---

### 🔹 Why Multi-level U-Net?

- **Single U-Net Limitation:** Misses small or low-contrast tumors.  
- **Multi-level Advantage:** By cropping to liver ROI, the second model focuses only on meaningful regions → **higher Dice score on tumors (+5~10%)**.  
- **Clinical Impact:** Structured reports with tumor **count, size, and location** → radiologists save **~60% review time**.


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
