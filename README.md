# Hierarchical Region-Context Attention for Image Captioning (HRcAIC) — EAAI 2026

This repository contains the official implementation of **Hierarchical Region-Context Attention for Image Captioning (HRcAIC)**, published in **Engineering Applications of Artificial Intelligence**, Volume **168** (15 March 2026), Article **114014**.

HRcAIC is a hybrid encoder–decoder framework that explicitly integrates **object-level region features** and **global scene context** using a **Region-Context Attention Network (RCAN)**, followed by **Hierarchical Attention-Based (HAB) context encoding** with **spatial + channel attention**, and a **hierarchical LSTM decoder** for fluent caption generation.

**Paper DOI:** https://doi.org/10.1016/j.engappai.2026.114014  
**Code:** https://github.com/alamgirustc/HRcAIC

---

## 📌 Highlights

- **Region–Context fusion (RCAN):** multi-head attention fuses Faster R-CNN region features with ResNet global context.
- **HAB context encoding:** hierarchical refinement with **spatial attention + channel attention**.
- **Hierarchical LSTM decoder:** efficient autoregressive caption generation (linear in caption length).
- **Strong results on MS COCO 2014 (Karpathy split):**
  - **BLEU-4 = 40.0**, **CIDEr = 132.5**, **SPICE = 23.6** (CIDEr-optimized / RL).

---

## 🖼️ Framework Overview

<p align="center">
  <img src="images/framework_hrcaic.jpg" width="900"/>
</p>

**Figure:** HRcAIC architecture (RCAN + HAB + hierarchical decoder).  
> Put your final figure at: `images/framework_hrcaic.jpg`

---

## 🧾 Citation

If you use this code or build upon our work, please cite:

```bibtex
@article{hossain2026hrcaic,
  title={Hierarchical Region-Context Attention for image captioning},
  author={Hossain, Mohammad Alamgir and Ye, ZhongFu and Hossen, Md. Bipul and Rahman, Md. Atiqur and Islam, Md Shohidul and Abdullah, Md. Ibrahim},
  journal={Engineering Applications of Artificial Intelligence},
  volume={168},
  pages={114014},
  year={2026},
  doi={10.1016/j.engappai.2026.114014}
}
```

### Baseline reference (X-LAN)
We also recommend citing the key baseline used in comparisons:

```bibtex
@inproceedings{xlinear2020cvpr,
  title={X-Linear Attention Networks for Image Captioning},
  author={Pan, Yingwei and Yao, Ting and Li, Yehao and Mei, Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

---

## 🔧 Requirements

- Python 3.8+ (recommended)
- CUDA 10+ (or CUDA 11+ depending on your PyTorch build)
- PyTorch >= 1.10 (recommended)
- torchvision
- numpy, tqdm, easydict
- coco-caption (for COCO evaluation)

Install core dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Data Preparation (MS COCO 2014)

HRcAIC follows the standard MS COCO 2014 pipeline using region-level features (Bottom-Up style) and caption annotations.

### 1) COCO Captions and Karpathy Split
- Download MS COCO 2014 images and captions.
- Ensure you are using the **Karpathy split** (train/val/test = 113,287 / 5,000 / 5,000).

Organize (example):
```
data/
  coco/
    images/
      train2014/
      val2014/
    annotations/
      captions_train2014.json
      captions_val2014.json
    karpathy_split/
      dataset_coco.json
```

### 2) Bottom-Up Region Features (Faster R-CNN)
Use Faster R-CNN (ResNet-101 backbone, pretrained on Visual Genome), then convert features to `.npz`.

If you use TSV-style features (Bottom-Up Attention style), convert them:

```bash
python2 tools/create_feats.py \
  --infeats bottom_up_tsv \
  --outfolder ./data/coco/features/up_down_10_100
```

> If your repo already includes a different extractor (recommended), document it here and provide a script such as:
> `tools/extract_frcnn_features.py`

### 3) Global Context Features (ResNet-101)
Extract global features from ResNet-101 (full image), store as `.npz` or `.pth`:

```bash
python tools/extract_global_resnet101.py \
  --image_root ./data/coco/images \
  --out ./data/coco/features/global_resnet101.npz
```

> Replace with your actual script name/flags.

### 4) COCO Evaluation Toolkit
Clone and configure `coco-caption`:

```bash
git clone https://github.com/ruotianluo/coco-caption.git
```

Set the path in your config file (example):
- `lib/config.py` → `C.INFERENCE.COCO_PATH = "/path/to/coco-caption"`

---

## 🏋️ Training

HRcAIC training typically has two stages:
1) Cross-Entropy (CE) pretraining  
2) Reinforcement Learning (SCST) fine-tuning with CIDEr reward

### Train with Cross-Entropy (CE)

```bash
bash experiments/hrcaic/train_ce.sh
```

### Train with SCST (CIDEr Optimization)

Copy your best CE checkpoint into the RL snapshot folder:

```bash
cp experiments/hrcaic/snapshot/model_best.pth experiments/hrcaic_rl/snapshot/
bash experiments/hrcaic_rl/train_rl.sh
```

---

## ✅ Evaluation

Evaluate a trained model (beam search or greedy):

```bash
CUDA_VISIBLE_DEVICES=0 python main_test.py \
  --folder experiments/hrcaic \
  --resume model_best
```

Outputs:
- Captions JSON
- COCO metrics (BLEU, METEOR, ROUGE-L, CIDEr, SPICE)

---

## 📊 Reproducing Paper Results

We report (Karpathy test split):

### Cross-Entropy (CE)
- BLEU-4: 38.3
- CIDEr: 122.1
- SPICE: 22.1

### CIDEr-Optimized (RL / SCST)
- BLEU-4: 40.0
- CIDEr: 132.5
- SPICE: 23.6

> To reproduce, keep the backbone fixed (Faster R-CNN + ResNet-101), same split, same beam size, and same preprocessing.

---

## 🧠 Model Components

### RCAN (Region-Context Attention Network)
- Query: RoI region features
- Key/Value: global context features
- Multi-head attention + residual fusion for stable training

### HAB (Hierarchical Attention-Based Context Encoding)
- Spatial attention: emphasizes informative regions
- Channel attention: emphasizes semantically rich feature channels
- Element-wise fusion for robust context vectors

### Hierarchical Decoder
- Two-stage LSTM
- Visual attention guided by HAB-refined tokens
- GLU-based fusion for controlled injection of visual evidence

---

## 📁 Pretrained Models

You may download pretrained checkpoints here:

- **CE model:** (add link)
- **RL model:** (add link)

Example:
```
experiments/hrcaic/snapshot/model_best.pth
experiments/hrcaic_rl/snapshot/model_best.pth
```

---

## 🙏 Acknowledgements

This repository is inspired by and/or builds upon:
- **Bottom-Up and Top-Down Attention** (Anderson et al., 2018)
- **Self-Critical Sequence Training (SCST)** (Rennie et al., 2017)
- **X-LAN / X-Transformer** (Pan et al., 2020)
- COCO caption evaluation code: `coco-caption`

---

## 📌 License

Add your preferred license here (e.g., MIT, Apache-2.0, etc.).

---

## ✉️ Contact

For questions or collaborations:

- **Mohammad Alamgir Hossain**
- GitHub: https://github.com/alamgirustc
