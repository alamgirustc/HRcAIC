# HRcAIC: Hierarchical Region-Context Attention for Image Captioning (EAAI 2026)

This repository contains the official implementation of **Hierarchical Region-Context Attention for Image Captioning (HRcAIC)**, published in **Engineering Applications of Artificial Intelligence (2026)**. :contentReference[oaicite:1]{index=1}

HRcAIC improves caption quality by **explicitly fusing object-level region features with global scene context** using a **Region-Context Attention Network (RCAN)**, then refining the fused representation using **Hierarchical Attention-Based (HAB) context encoding** (spatial + channel attention), and generating captions with a **hierarchical LSTM decoder**. :contentReference[oaicite:2]{index=2}

---

## Paper

**Hierarchical Region-Context Attention for image captioning**  
Mohammad Alamgir Hossain, ZhongFu Ye, Md. Bipul Hossen, Md. Atiqur Rahman, Md Shohidul Islam, Md. Ibrahim Abdullah  
Engineering Applications of Artificial Intelligence, Volume 168, 2026, 114014  
DOI: https://doi.org/10.1016/j.engappai.2026.114014 :contentReference[oaicite:3]{index=3}

### Please cite (BibTeX)

```bibtex
@article{HOSSAIN2026114014,
title = {Hierarchical Region-Context Attention for image captioning},
journal = {Engineering Applications of Artificial Intelligence},
volume = {168},
pages = {114014},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2026.114014},
url = {https://www.sciencedirect.com/science/article/pii/S0952197626002952},
author = {Mohammad Alamgir Hossain and ZhongFu Ye and Md. Bipul Hossen and Md. Atiqur Rahman and Md Shohidul Islam and Md. Ibrahim Abdullah},
keywords = {Image captioning, Region-Context Attention, Hierarchical attention, Visual fusion, Caption generation},
abstract = {Image captioning is a challenging task that requires a deep understanding of both visual and linguistic modalities to generate accurate and meaningful descriptions. Traditional methods often struggle to effectively integrate object-level and global scene features, leading to limited contextual awareness in generated captions. To address this, we propose a novel Hierarchical Region-Context Attention for Image Captioning framework that combines a Region-Context Attention Network for multi-scale visual feature fusion with a Hierarchical Attention-Based context encoding mechanism for refined representation learning. The Region-Context and Hierarchical Attention module extracts object-level features using Faster Region-based Convolutional Neural Network and global context features from Residual Networks, integrating them through a multi-head attention mechanism. This fusion enables localized object representations to be enriched with scene-level semantics. The fused visual features are further refined using a hierarchical attention-based approach, which employs both spatial and channel-wise attention to emphasize semantically relevant information across regions and dimensions. The decoder is implemented using a hierarchical Long Short-Term Memory network that generates captions in an autoregressive manner, leveraging the hierarchical attention-based refined features to guide each word prediction. This structure enables the model to maintain temporal coherence while dynamically attending to informative visual content. We evaluate our model on the Microsoft Common Objects in Context 2014 dataset, achieving a Bilingual Evaluation Understudy score of 40.0 and a Consensus-based Image Description Evaluation score of 132.5, surpassing state-of-the-art models. Results indicate that the model effectively captures object details and context, producing more coherent and accurate captions. The code for this project is publicly available at https://github.com/alamgirustc/HRcAIC.}
}
