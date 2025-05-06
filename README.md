#  Source Printer Identification using PSLTD and Component Feature Pooling

This project implements a complete computer vision pipeline for **source printer identification** using connected component analysis, feature extraction, and classification. The method is based on a combination of **Printer Specific Local Texture Descriptor (PSLTD)** and Support Vector Classifiers (SVCs) as described in leading research on this topic.

---

## Project Summary

Printed documents contain subtle texture patterns introduced by the printer's hardware. This project detects and extracts these patterns to classify the **source printer** of a scanned page.

The pipeline involves:

1. **Connected Component Extraction** (isolating characters/letters)
2. **Feature Extraction** (using PSLTD and other texture descriptors)
3. **Pooling** (aggregating features across all characters)
4. **Classification** (via 15 parallel SVCs trained on pooled features)

---

## 📁 Repository Structure

| File/Folder                       | Description                                                    |
| --------------------------------- | -------------------------------------------------------------- |
| `ConnectedComponentExtraction.py` | Detects individual characters or blobs from scanned images     |
| `ComponentFeatureExtractor.py`    | Extracts PSLTD and other feature vectors from each component    |
| `ColumnPooling.py`                | Pools features across components for page-level representation |
| `main.py`                         | Executes full pipeline: extraction → pooling → classification  |
| `requirements.txt`                | Lists required Python packages                                 |

---

## PSLTD: Printer Specific Local Texture Descriptor

A core strength of this project is its use of **PSLTD**, a hand-crafted descriptor developed for identifying the source of a printed document. PSLTD captures tiny distortions, ink distributions, and printer-specific textures that are not visible to the naked eye.

### What Makes PSLTD Special?

* **Local Texture Focus**: Extracts fine-grained micro-patterns from small image patches.
* **Printer Discriminative**: Tailored to distinguish between printers, not just general textures.
* **Works Without OCR**: Uses binary image patches from character blobs directly.

### How PSLTD Is Used?

1. **Component Extraction**:

   * Letters are detected using connected component analysis.
   * Each component is binarized and normalized for feature computation.

2. **Feature Extraction**:

   * For each component, PSLTD features are extracted to capture geometric and texture characteristics.
   * Total of **2 feature vector** per component.

3. **Pooling**:

   * Component-level features are aggregated using average pooling column-wise to create a **page-level feature vector per column descriptor**.

4. **Classification**:

   * Each of the 15 columns is used to train a separate **Support Vector Classifier (SVC)**.
   * Final classification is based on **majority voting** or **score fusion** across all classifiers.

---

##  Getting Started

### Prerequisites

* Python 3.7+
* Recommended: OpenCV, NumPy, Scikit-learn

Install required packages:

```bash
pip install -r requirements.txt
```

### Input

* Scanned printed pages (preferably grayscale or binarized)
* Recommended resolution: **300-600 dpi**

### Run the Full Pipeline

```bash
python main.py
```

This will:

* Load scanned images
* Extract connected components
* Compute PSLTD & other features
* Pool features
* Train and test 15 SVC classifiers

---

## Key References

* **[Source Printer Identification using Printer Specific Pooling of Letter Descriptors](https://arxiv.org/abs/2109.11139)**
  *Sharad Joshi, Yogesh Kumar Gupta, Nitin Khanna — 2021*

* **[Source Printer Classification using Printer Specific Local Texture Descriptor](https://arxiv.org/abs/1806.06650)**
  *Sharad Joshi, Nitin Khanna — 2018*

---

## Contributors

* Implemented by \[Suprabho Saha, Rahul, Anant Yadav, Manchi Vikranth]
