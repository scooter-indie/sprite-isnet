# Sprite IS-Net: Custom Background Removal for Sprite Sheets

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Export-green.svg)](https://onnx.ai/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Train custom IS-Net models for precise sprite sheet background removal** and integrate them with [rembg](https://github.com/danielgatis/rembg) for production use.

Built on top of the [DIS (Dichotomous Image Segmentation)](https://github.com/xuebinqin/DIS) architecture, this project provides a complete Windows-based workflow for training, converting, and deploying custom background removal models optimized for game sprites, pixel art, and character sheets.

---

## 🎯 Features

- **🎮 Sprite-Optimized Training**: Fine-tune IS-Net models specifically for sprite sheets and game assets
- **🚀 Complete Windows Workflow**: Automated setup scripts from environment to deployment
- **🔄 ONNX Export**: Convert trained models to ONNX format for production use
- **🔗 Rembg Integration**: Seamlessly integrate custom models with rembg CLI and API
- **📊 Training Utilities**: Dataset preparation, validation, and quality checking tools
- **⚡ GPU Accelerated**: CUDA support for fast training and inference
- **📦 Batch Processing**: Process entire directories of sprite sheets efficiently

---
