# HRM-ReRead-MT: Hierarchical Re-Reading with Multi-Teacher Latent Consolidation

**A theoretical framework and complete implementation for data-efficient LLM training**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv-like Paper](https://img.shields.io/badge/Paper-PDF-red)](https://github.com/AJSubrizi/HRM-ReRead-MT/blob/main/paper.pdf)

## Overview

**HRM-ReRead-MT** extends the brain-inspired Hierarchical Reasoning Model (HRM-Text, Sapient Intelligence 2026) with two novel ideas:

- **Iterative Re-Reading** with latent-state consolidation (inspired by human hippocampal replay)
- **Multi-Teacher Distillation** directly from DeepSeek-V4-Pro into the high-level latent module

The goal is to achieve significantly higher reasoning depth and data-efficiency by letting a small model (1B parameters) **progressively consolidate knowledge** across multiple passes over high-quality data, guided by a much larger teacher.

This repository contains:
- Full theoretical paper (LaTeX + PDF)
- Complete training pipeline (data augmentation + Re-Read training)
- Ready-to-run scripts
- All code needed to reproduce the framework

**Empirical validation is pending** (compute constraints). We release everything openly to invite collaboration and testing.

## Paper

📄 **Full Paper**: [PAPER.md](PAPER.md)  
📑 **Compiled PDF**: [paper.pdf](paper.pdf) (LaTeX source also included)

**Title**:  
HRM-ReRead-MT: Hierarchical Re-Reading with Multi-Teacher Latent Consolidation for Data-Efficient LLM Training

**Author**: Antonio J. Subrizi

## Key Features

- Hierarchical recurrent architecture (HRM backbone)
- 5 progressive Re-Reading epochs with latent memory buffer
- KL-divergence + hidden-state distillation from DeepSeek-V4-Pro
- Theoretical projected gains of +15–26 points on major reasoning benchmarks
- Fully documented and MIT licensed

## Repository Structure

HRM-ReRead-MT/
├── PAPER.md                  # Full scientific paper (Markdown)
├── paper.pdf                 # Compiled LaTeX version
├── main.tex                  # LaTeX source
├── LICENSE                   # MIT License
├── README.md                 # This file
├── HRM-Text/                 # Original HRM-Text repo (cloned)
├── data_augment.py           # Augment dataset with DeepSeek-V4-Pro
├── train_re_read_mt.py       # Core training loop (ReRead + distillation)
├── teacher_utils.py          # Teacher API utilities
├── setup.sh                  # One-click environment setup
└── create_hrm_reread_mt.py   # Script that generated the whole package

## Quick Start (for testing / collaboration)

1. Clone the repo:
   ```bash
   git clone https://github.com/AJSubrizi/HRM-ReRead-MT.git
   cd HRM-ReRead-MT
   Get your DeepSeek API key and put it in .env
   Run the full pipeline (data + training):
   bash setup.sh
python data_augment.py
torchrun --nproc_per_node=8 train_re_read_mt.py   # (requires 8×H100 or equivalent)
Note: Full training requires large GPU clusters. On consumer hardware you can run only the data augmentation step.
