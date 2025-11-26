# CMAF-DDI: Cross-Modal Attention Fusion for Multi-Class Drug-Drug Interaction Prediction

[![Framework](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

This repository contains the official PyTorch implementation of the paper: **"CMAF-DDI: Multi-Class Drug-Drug Interactions Prediction Method Based on Cross-Modal Attention Fusion"**.

## 📖 Introduction

**CMAF-DDI** is a novel deep learning framework designed to predict multi-class Drug-Drug Interactions (DDIs). Unlike traditional methods, CMAF-DDI integrates three distinct modalities to capture comprehensive drug features:
* **Protein Sequence Features** (Target information)
* **Molecular Graph Features** (Chemical structure)
* **Knowledge Graph Features** (Semantics and relations)

To effectively fuse these modalities, we propose a two-level **Cross-Modal Attention Fusion** mechanism that adaptively highlights critical features (e.g., functional groups or binding sites) relevant to specific interaction types.

## 📂 Project Structure

```text
CMAF-DDI/
├── data/                  # Raw data files (e.g., DRKG, drug lists)
├── Knowledge Graph/       # Scripts/Data for KG embedding processing
├── Molecular Graph/       # Scripts/Data for Molecular Graph processing
├── Protein sequence/      # Scripts/Data for Protein Sequence processing
├── models/                # Core model definitions (CMAF architecture)
├── utils/                 # Utility functions (metrics, data loaders)
├── main.py                # Main entry point for training and testing
├── requirements.txt       # Python dependencies
└── README.md              # This file
