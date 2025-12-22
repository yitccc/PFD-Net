# PFD-Net: A Physically Informed and Frequency-Interactive Density-Aware Cross-Modal Network for Forest Fire Detection
Reliable forest fire detection in complex scenarios remains a significant challenge due to the inherent limitations of single-modality remote sensing sensors: visible (VIS) images are susceptible to dense smoke occlusion, while infrared (IR) images suffer from semantic ambiguity and false positives caused by non-fire hot objects. Existing multimodal fusion methods often lack explicit physical guidance and struggle to balance detection accuracy with inference efficiency. To address these issues, we propose PFD-Net, a physically informed and frequency-interactive density-aware cross-modal network. First, the Smoke-Prior Guided Alternating Alignment Module (SPG-AAM) embeds a differentiable dark-channel prior to explicitly enhance smoke representation and boundaries. It aligns VIS and IR features in an alternating manner to effectively recover occluded flame targets. Second, the Phase-Differentiated Modal Interaction Module (PDMIM) operates in the frequency domain. Leveraging the insight that phase encodes structural topology while amplitude captures semantic strength, PDMIM utilizes phase discrepancy to transfer structural information across modalities without compromising their distinct intensity statistics. Third, we introduce a density-guided adaptive mechanism comprising a Density-aware Feature Extractor (DaFE) and Prior-Adaptive Query Inference (PAQI). This mechanism dynamically predicts a target density map and adjusts the query budget, efficiently handling sparse scenes without computational redundancy while ensuring sufficient coverage for dense flame clusters. Extensive experiments on the FLAME 2 and FLAME 3 datasets demonstrate the superiority of PFD-Net. It achieves a state-of-the-art mAP50 of 63.7\% on FLAME 2, surpassing existing fusion methods while maintaining a lightweight parameter count of 30.9 M and a real-time inference speed of 23.6 ms, making it highly suitable for UAV-based wildfire monitoring.
## Datasets

```
├── dataset
│   ├── image
│   │   ├── train
|   |   ├── val
│   ├── images
│   │   ├── train
|   |   ├── val
│   ├── labels
│   │   ├── train
|   |   ├── val
```
