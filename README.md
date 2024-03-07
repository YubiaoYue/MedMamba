# MedMamba: Vision Mamba for Medical Image Classification
Medical image classification is one of the most important tasks in computer vision and serves as the foundation for other advanced tasks, such as medical object detection and medical image segmentation. Inspired by the visual state space model, we propose Vision Mamba for medical image classification. To demonstrate the potential of MedMamba, we conduct extensive experiments using three publicly available medical datasets with different imaging techniques (i.e., Kvasir (endoscopic images), FETAL_PLANES_DB (ultrasound images) and Covid19-Pneumonia-Normal Chest X-Ray (X-ray images)) and two private datasets built by ourselves. Experimental results show that the proposed MedMamba performs well in detecting lesions in various medical images. To the best of our knowledge, this is the first Vision Mamba tailored for medical image classification. The purpose of this work is to establish a new baseline for medical image classification tasks and provide valuable insights for the future development of more efficient and effective SSM-based artificial intelligence algorithms and application systems in the medical.
![Medmamba](https://github.com/YubiaoYue/MedMamba/assets/141175829/12f9def3-38c2-46b2-bdf8-c090d18e436e)
# Installation
1. Install the package:  ```bash  $ pip install causal-conv1d>=1.1.0
2. pip install mamba-ssm: the core Mamba package.
Other requirements:
1. lINUX
2. NVIDIA GPU
3. PyTorch 1.18+
4. CUDA 11.8+
# Datasets
## Kavsir
