# Multi-Stream Deep Learning & Error Level Analysis (ELA) for Image Forgery Detection

A forensic research framework and full-stack software system designed to detect digital image manipulations (splicing, copy-move, inpainting). This project combines a high-performance **Rust-powered Error Level Analysis (ELA) engine**, a **dual-stream Deep Residual Neural Network (Late Feature Fusion)**, and an interactive **React/TypeScript analytic dashboard**.

---

## Key Highlights & Research Focus

* **Dual-Stream Forensic Architecture**: Combines raw spatial RGB representations with frequency/compression artifact residuals (ELA) using a dual ResNet backbone with late feature fusion.
* **Optimized ELA Engine (Rust)**: High-throughput computation of recompression residuals, quantization matrix alignment, and differential pixel sweeps.
* **Compression Robustness Evaluation**: Extensive robustness testing across JPEG compression sweeps ($Q = [50, 75, 85, 90, 100]$) to address ELA sensitivity degradation.
* **Full-Stack Forensic Workbench**: Interactive client application supporting dynamic ELA factor calibration, anomaly inspection overlays, and real-time inference confidence visualization.

---

## System Architecture

```
                       +---------------------------+
                       |    Input Target Image     |
                       +-------------+-------------+
                                     |
                 +-------------------+-------------------+
                 |                                       |
                 v                                       v
         [ RGB Spatial Stream ]                 [ Forensic ELA Stream ]
                 |                                       |
                 |                                  Rust Engine
                 |                           (Quantization & Residuals)
                 v                                       v
         [ ResNet Backbone ]                     [ ResNet Backbone ]
                 |                                       |
                 +-------------------+-------------------+
                                     |
                                     v
                       [ Late Feature Fusion Layer ]
                                     |
                                     v
                         Binary Classification Output
                     (Authentic vs. Forged Probability)

```

---

## Repository Structure

```text
.
├── Client/
│   └── react-client/              # Interactive forensic web workbench
│       ├── src/
│       │   ├── components/        # Layout, ActiveViewer, AnalyticPanel, DragDrop
│       │   ├── styles/            # Modular SCSS & BEM styling
│       │   ├── utils/             # Frontend ELA processors & helpers
│       │   └── types/             # Domain TypeScript definitions
│       └── package.json
│
└── ela_core/                      # Core ML models, Rust engine, and evaluation suite
    ├── src/                       # High-performance Rust ELA implementation (lib.rs)
    ├── services/                  # FastAPI/Python inference server & core forensic logic
    ├── fusion/                    # Dual-stream Late Feature Fusion training pipelines
    ├── tests/
    │   ├── data_integrity/        # Train/Val leakage and distribution validators
    │   ├── ela_methods/           # SRM vs. ELA comparison & quantization metrics
    │   ├── model_evaluation/      # Baseline ROC-AUC, Precision-Recall, Confusion matrices
    │   └── robustness/            # JPEG quality sweep stress tests (N=200, N=1000)
    ├── artifacts/                 # Generated ROC curves, evaluation manifests, and plots
    ├── Cargo.toml                 # Rust crate configuration
    └── pyproject.toml             # Python package configuration

```

---

## Methodology & Experimental Results

### 1. Dual-Stream Late Feature Fusion

Traditional RGB neural networks frequently overfit to high-level semantic content rather than subtle digital tampering traces. Conversely, pure ELA maps often produce high false-positive rates on complex spatial textures.

This framework resolves this trade-off by concatenating dense feature embeddings from both streams before final classification, enabling the model to balance semantic context and compression inconsistencies.

### 2. Forensic Evaluation & Benchmarks

The pipeline was evaluated across standard digital forensics datasets (including CASIA 2.0 and synthetic splicing sets):

| Model Variant | Feature Representation | Robustness ($Q \ge 75$) | Primary Detection Vector |
| --- | --- | --- | --- |
| **RGB Baseline** | Raw Pixel Tensor | Moderate | Spatial seams, edge discrepancies |
| **ELA Baseline** | Compression Residual | High | Quantization table mismatches |
| **Fusion Network** | **RGB + ELA Late Fusion** | **Superior** | **Joint Spatial-Frequency Anomalies** |

Evaluation artifacts located in `ela_core/artifacts/images/plots/` provide detailed:

* Receiver Operating Characteristic (ROC-AUC) curves.
* Recompression sweep manifests demonstrating stability under aggressive JPEG downsampling.
* Confusion matrices at threshold-optimized operating points.

---

## Quick Start

### Prerequisites

* **Node.js**: v18+ and `npm`
* **Python**: 3.10+
* **Rust**: `cargo` and `rustc` (latest stable)

---

### Backend Service (`ela_core`)

1. Navigate to the core service directory:
```bash
cd ela_core

```


2. Create and activate a Python virtual environment:
```bash
python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

```


3. Install required Python packages and build the native module:
```bash
pip install -r requirements.txt  # or: pip install .
maturin develop --release        # Builds optimized native Rust bindings

```


4. Launch the local inference server:
```bash
python -m services.server

```



---

### Frontend Client (`Client/react-client`)

1. Open a separate terminal and navigate to the UI directory:
```bash
cd Client/react-client

```


2. Install dependencies:
```bash
npm install

```


3. Start the development server:
```bash
npm run dev

```


4. Open `http://localhost:5173` in your browser to inspect images and run inference.

---

## Academic & Degree Context

* **Project Title**: Dual-Stream Image Forgery Detection via Error Level Analysis and Deep Residual Learning
* **Author**: Oleksandr Fedorov
* **Degree**: Master's Thesis / Research Diploma Work
* **Primary Scope**: Computer Engineering, Computer Vision, Digital Forensics, High-Performance Systems
