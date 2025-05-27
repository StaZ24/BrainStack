# BrainStack

> Functionally Guided Meta-Ensemble Learning for EEG-Based Neural Decoding  
> [NeurIPS 2025 Submission]  
> Anonymous authors • Code and data will be released upon acceptance

## 🔬 Overview
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/) [![NeurIPS 2025 Submission](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)

**BrainStack** is a functionally guided, heterogeneous ensemble framework for EEG-based text decoding. Inspired by neuroscience, it partitions EEG signals by brain region and fuses local and global neural representations using a gated meta-learner.

This repository contains the code and instructions to reproduce the experiments in our NeurIPS 2025 submission.

<div align="center">
  <img src="assets/brainstack_architecture.png" width="600"/>
  <p><i>Fig: BrainStack architecture with global CTNet, local CNNs, and a Gated Meta-Learner.</i></p>
</div>

---

## 🚀 Highlights

- 🧩 **Heterogeneous Architecture**: Combines a global Transformer encoder (CTNet) with seven lightweight regional CNNs (CNet).
- 🎯 **Gated Expert Fusion**: Meta-learner adaptively fuses region-wise features with learnable attention weights.
- 🔁 **Knowledge Distillation**: The global branch guides regional experts to align local semantics.
- 📊 **New Dataset**: Introduces SS-EEG, one of the largest silent speech EEG datasets (120+ hours, 10 subjects, 24-word vocabulary).
- 📈 **SOTA Performance**: Achieves 41.87% avg. accuracy on a 24-way classification task, surpassing CNN/Transformer baselines and pretrained models.

---

## 📁 Project Structure

```
BrainStack/
├── configs/                # YAML configuration files for different experiments
├── data/                   # Data preprocessing and loading scripts (no raw EEG included)
├── models/                 # CTNet, CNet, Gated Meta-Learner implementations
├── loss/                   # Dynamic multi-objective loss and distillation
├── trainer/                # Training, evaluation, and logging scripts
├── utils/                  # Miscellaneous utilities (e.g., metrics, schedulers)
├── assets/                 # Images for README and paper
├── main.py                 # Entry point for training and evaluation
├── inference.py            # Run inference with trained model
└── requirements.txt        # Required Python packages
```

---

## 🧠 Dataset

We introduce **SilentSpeech-EEG (SS-EEG)**, a 120-hour EEG dataset for silent speech decoding across 12 subjects.  

| Feature         | Value                 |
|----------------|-----------------------|
| Subjects        | 10 (final release)    |
| Words           | 24 (6 semantic classes) |
| Trials / Subject | 6000                 |
| Duration        | 120+ hours total      |
| Channels        | 122 EEG + 11 extras   |
| Sampling Rate   | 1000 Hz               |


> 📌 *Due to ethics restrictions, anonymized EEG recordings will be released upon request after publication.*

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/anonymous/BrainStack.git
cd BrainStack
```

### 2. Create a conda environment

```bash
conda create -n brainstack python=3.8
conda activate brainstack
pip install -r requirements.txt
```

### 3. Run training (example)

```bash
python main.py 
```

---

## 📈 Performance


| Model             | Params | Avg. Acc (%) |
|------------------|--------|---------------|
| EEGNet            | 8.5K   | 28.78         |
| TCNet             | 78K    | 29.50         |
| EEGConformer      | 0.75M  | 23.89         |
| BrainStack (Ours) | 1.06M  | **41.87**     |

See [`results/`](results/) for detailed logs and plots.

---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@article{brainstack2025,
  title={BrainStack: Functionally Guided Meta-Ensemble Learning for EEG-Based Neural Decoding},
  author={Anonymous},
  journal={NeurIPS 2025 Submission},
  year={2025},
  url={https://BtrrJL24.github.io/BrainStack}
}
```
---
## 📄 License

The code is released under the **CC BY-NC 4.0 License** – free for academic and research use.

---

## 📬 Contact

For any questions or collaborations, feel free to open an issue or contact the authors (details to appear after review).

---