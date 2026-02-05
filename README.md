# HRcAIC — Hierarchical Region-Context Attention for Image Captioning

This repository provides an implementation of **HRcAIC** (Hierarchical Region-Context Attention for Image Captioning).

**Paper:** *Hierarchical Region-Context Attention for image captioning*, Engineering Applications of Artificial Intelligence, **168** (2026) 114014.  
**DOI:** 10.1016/j.engappai.2026.114014

---

## Highlights
- **Region–Context Attention Network (RCAN)** fuses object-level region features and global scene context using **multi-head attention**.
- **Hierarchical Attention-Based (HAB) context encoding** refines fused features via **spatial** and **channel-wise** attention.
- **Hierarchical LSTM decoder** generates captions with improved grounding and fluency.

---

## Requirements
* [PyTorch](http://pytorch.org/) (>1.0)
* [torchvision](http://pytorch.org/)
* [coco-caption](https://github.com/ruotianluo/coco-caption)

A typical environment:
- Python 3.7+
- CUDA (optional but recommended)
- Common Python packages: numpy, tqdm, h5py, scipy, pyyaml, tensorboardX (as needed)

> Note: Some feature-conversion utilities use **python2** (see Data preparation step 1).

---

## Installation
```bash
git clone https://github.com/alamgirustc/HRcAIC.git
cd HRcAIC

# create and activate env (example)
conda create -n hrcaic python=3.8 -y
conda activate hrcaic

pip install -r requirements.txt
```

---

## Data preparation

1. Download the [bottom up features](https://github.com/peteanderson80/bottom-up-attention) and convert them to npz files
```bash
python2 tools/create_feats.py --infeats bottom_up_tsv --outfolder ./mscoco/feature/up_down_10_100
```

2. Download the [annotations](https://drive.google.com/open?id=1i5YJRSZtpov0nOtRyfM0OS1n0tPCGiCS) into the `mscoco/` folder.  
   More details about data preparation can be referred to [self-critical.pytorch](https://github.com/ruotianluo/self-critical.pytorch)

3. Download [coco-caption](https://github.com/ruotianluo/coco-caption) and set `__C.INFERENCE.COCO_PATH` in `lib/config.py` to your local `coco-caption` path.

4. Download the trained checkpoints and model-generated test captions:

- **Cross-Entropy (CE) checkpoint (Cross):**  
  https://drive.google.com/file/d/17hlrm0z6OUoz3OC_OPdKlfAy1z_W6JnB/view?usp=sharing

- **Cross-Entropy (CE) model-generated test captions (results):**  
  https://drive.google.com/file/d/1Lt6baHo-aQrb6m2-1k_48z8sqn8KoBc2/view?usp=sharing

- **CIDEr-Optimized (SCST/RL) checkpoint (RL):**  
  https://drive.google.com/file/d/1MXABJJH0q1l9PFwXmiz3cQGpqxUfEjeG/view?usp=sharing

- **CIDEr-Optimized (SCST/RL) model-generated test captions (results):**  
  https://drive.google.com/file/d/1EXGE6V_KzZmDfyyf3FjEn9QG9e7jODlF/view?usp=sharing

Suggested layout after extracting:
```text
checkpoints/
  ce/
    model.pth
  rl/
    model.pth
results/
  ce/
    test_captions.json
  rl/
    test_captions.json
```
(Adjust folder names/paths to match your config and scripts.)

---

## Project structure (typical)
- `tools/` — utilities (feature conversion, helpers)
- `mscoco/` — dataset assets (annotations, features, splits, cache)
- `lib/` — core modules (models, configs, training, inference)
- `configs/` — experiment configs (if provided)
- `scripts/` — launch/train/eval scripts (if provided)

(Names may vary slightly depending on your repo layout.)

---

## Training

### 1) Cross-entropy training (CE)
Train with teacher-forcing / cross-entropy objective.
```bash
# Example (adapt to your repo's entrypoints)
python train.py --cfg configs/coco_ce.yaml --save_dir checkpoints/ce
```

### 2) CIDEr-optimized training (SCST / RL)
Fine-tune with CIDEr-based reward using self-critical sequence training.
```bash
# Example (adapt to your repo's entrypoints)
python train.py --cfg configs/coco_rl.yaml --resume checkpoints/ce/best.pth --save_dir checkpoints/rl
```

> Keep your config consistent with the MS COCO Karpathy split and the feature folder from **Data preparation**.

---

## Evaluation
```bash
# Example (adapt to your repo's entrypoints)
python eval.py --checkpoint checkpoints/rl/best.pth --split test --beam_size 3
```
This typically reports BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr, and SPICE.

---

## Inference (caption generation)
```bash
# Example (adapt to your repo's entrypoints)
python infer.py --checkpoint checkpoints/rl/best.pth --image_path path/to/image.jpg --beam_size 3
```

---

## Citation
If you find this work useful, please cite:

```bibtex
@article{hossain2026hrcaic,
  title   = {Hierarchical Region-Context Attention for image captioning},
  author  = {Hossain, Mohammad Alamgir and Ye, ZhongFu and Hossen, Md. Bipul and Rahman, Md. Atiqur and Islam, Md Shohidul and Abdullah, Md. Ibrahim},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {168},
  pages   = {114014},
  year    = {2026},
  doi     = {10.1016/j.engappai.2026.114014},
  publisher = {Elsevier}
}
```

---

## Acknowledgements
- Bottom-Up Attention features: https://github.com/peteanderson80/bottom-up-attention  
- COCO caption evaluation: https://github.com/ruotianluo/coco-caption  
- Data preparation reference: https://github.com/ruotianluo/self-critical.pytorch  
- X-LAN (baseline/reference): https://github.com/JDAI-CV/image-captioning (X-LAN)
