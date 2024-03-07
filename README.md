# MedMamba: Vision Mamba for Medical Image Classification
This is the official code repository for "MedMamba: Vision Mamba for Medical Image Classification". [Arxiv Paper](https://arxiv.org/abs/2403.03849)
# Work Summary
Medical image classification is one of the most important tasks in computer vision and serves as the foundation for other advanced tasks, such as medical object detection and medical image segmentation. Inspired by the visual state space model, we propose Vision Mamba for medical image classification. To demonstrate the potential of MedMamba, we conduct extensive experiments using three publicly available medical datasets with different imaging techniques (i.e., Kvasir (endoscopic images), FETAL_PLANES_DB (ultrasound images) and Covid19-Pneumonia-Normal Chest X-Ray (X-ray images)) and two private datasets built by ourselves. Experimental results show that the proposed MedMamba performs well in detecting lesions in various medical images. To the best of our knowledge, this is the first Vision Mamba tailored for medical image classification. The purpose of this work is to establish a new baseline for medical image classification tasks and provide valuable insights for the future development of more efficient and effective SSM-based artificial intelligence algorithms and application systems in the medical.
![Medmamba](https://github.com/YubiaoYue/MedMamba/assets/141175829/12f9def3-38c2-46b2-bdf8-c090d18e436e)
# Installation
* `pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install timm==0.4.12`
* `pip install triton==2.0.0`
## Other requirements:
* Linux
* NVIDIA GPU
* PyTorch 1.18+
* CUDA 11.8+
# Datasets
## Kavsir
The data is collected using endoscopic equipment at Vestre Viken Health Trust (VV) in Norway. The VV consists of 4 hospitals and provides health care to 470.000 people. One of these hospitals (the Bærum Hospital) has a large gastroenterology department from where training data have been collected and will be provided, making the dataset larger in the future. Furthermore, the images are carefully annotated by one or more medical experts from VV and the Cancer Registry of Norway (CRN). The CRN provides new knowledge about cancer through research on cancer. It is part of South-Eastern Norway Regional Health Authority and is organized as an independent institution under Oslo University Hospital Trust. CRN is responsible for the national cancer screening programmes with the goal to prevent cancer death by discovering cancers or pre-cancerous lesions as early as possible.[Kavsir Dataset](https://datasets.simula.no/kvasir/ "Download it")![kavisr](https://github.com/YubiaoYue/MedMamba/assets/141175829/1ef639fd-97a3-49db-8f96-aa11a6d664e8)

# Citation
```bibtex
@misc{yue2024medmamba,
      title={MedMamba: Vision Mamba for Medical Image Classification}, 
      author={Yubiao Yue and Zhenzhang Li},
      year={2024},
      eprint={2403.03849},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
