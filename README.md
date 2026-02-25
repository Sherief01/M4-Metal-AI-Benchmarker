#  M4-Metal-AI-Benchmarker
**High-Performance Neural Network Training on Apple Silicon M4**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=for-the-badge&logo=tensorflow)
![Metal](https://img.shields.io/badge/Apple-Metal_GPU-black?style=for-the-badge&logo=apple)

## 📌 Overview
This project demonstrates the raw AI processing power of the **Apple M4 Chip**. It uses **TensorFlow Metal (PluggableDevice)** to offload neural network training tasks to the local GPU, showcasing hardware-accelerated Machine Learning without relying on cloud resources.

## 💻 Hardware Specifications
- **Device:** Mac Mini M4 (2026 Edition)
- **Architecture:** ARM64 (Apple Silicon)
- **Acceleration:** Metal Performance Shaders (MPS)

## 📊 Benchmark Results
Tested using the **MNIST Dataset** (60,000 images of handwritten digits).

| Metric | Performance |
| :--- | :--- |
| **Model Type** | Multi-Layer Perceptron (MLP) |
| **Epochs** | 5 |
| **Batch Size** | 64 |
| **Total Training Time** | **18.67 Seconds** 🔥 |
| **Device Load** | High-Intensity Metal GPU Acceleration |



## 🛠️ Tech Stack & Skills Demonstrated
- **Frameworks:** TensorFlow (MacOS & Metal plugins)
- **Performance Engineering:** GPU-accelerated training workflows.
- **Environment Management:** Virtual Environment (venv) isolation for ARM64 architecture.
- **Version Control:** Professional Git workflow (Atomic commits, Token-based authentication).

## 🚀 How to Run Locally
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/Sherief01/M4-Metal-AI-Benchmarker.git](https://github.com/Sherief01/M4-Metal-AI-Benchmarker.git)
   cd M4-Metal-AI-Benchmarker
   
2.**Setup Environment:**
python3 -m venv venv
source venv/bin/activate
pip install tensorflow-macos tensorflow-metal pandas matplotlib

3.**Run Benchmark:**
python m4_benchmarker.py

Maintained by Sherief01 Building God-Level AI Agents & Cyber Security Architectures.

