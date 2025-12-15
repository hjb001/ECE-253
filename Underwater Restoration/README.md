# Real-World Image Degradations Restoration Project (ECE 253)

This repository contains the source code and results for the ECE 253 Group Project. Our work addresses three domains: Real Noise Denoising, JPEG Artifacts Reduction, and **Underwater Image Enhancement**.

## 📂 Project Structure

- `paper_DifferentBases_v3.ipynb`: The main Jupyter Notebook for **Underwater Image Enhancement**. It implements the multi-scale fusion pipeline and compares different wavelet bases (Haar, db4, sym6, and DT-CWT).
- `requirements.txt`: List of Python dependencies required to run the code.
- `*.jpg` / `test_images/`: Sample underwater images used for testing and validation.

## 👥 Team Members

- **Junbo Huang** (A69032266)
- **Ziming Xu** (A69031906)
- **Haifan Zhao** (A69041804)

---

## 🌊 Underwater Image Enhancement Section

This module focuses on restoring underwater images using a wavelet-based multi-scale fusion approach. We explicitly evaluate the performance of the **Dual-Tree Complex Wavelet Transform (DT-CWT)** against standard Discrete Wavelet Transforms (DWT).

### Prerequisites

To run the notebook, you need the following Python libraries. You can install them via pip:

```bash
pip install -r requirements.txt
