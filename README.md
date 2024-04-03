# MedMamba: Vision Mamba for Medical Image Classification
This is the official code repository for "***MedMamba: Vision Mamba for Medical Image Classification***"[![arXiv](https://img.shields.io/badge/arXiv-2403.03849-brightgreen.svg)](https://arxiv.org/abs/2403.03849)
![logo](https://github.com/YubiaoYue/MedMamba/assets/141175829/f55b3a61-26ab-4256-8fa9-a8f0022c63a2)
# Work Summary
Medical image classification is one of the most important tasks in computer vision and serves as the foundation for other advanced tasks, such as ***medical object detection*** and ***medical image segmentation***. Inspired by the visual state space model, we propose Vision Mamba for medical image classification. To demonstrate the potential of MedMamba, we conduct extensive experiments using three publicly available medical datasets with different imaging techniques (i.e., Kvasir (endoscopic images), FETAL_PLANES_DB (ultrasound images) and Covid19-Pneumonia-Normal Chest X-Ray (X-ray images)) and two private datasets built by ourselves. Experimental results show that the proposed MedMamba performs well in detecting lesions in various medical images. To the best of our knowledge, this is the first Vision Mamba tailored for medical image classification. ***The purpose of this work is to establish a new baseline for medical image classification tasks and provide valuable insights for the future development of more efficient and effective SSM-based artificial intelligence algorithms and application systems in the medical***.
![Medmamba](https://github.com/YubiaoYue/MedMamba/assets/141175829/d3affafa-2659-4625-bd01-08010358fca2)
![S6_BLOCK](https://github.com/YubiaoYue/MedMamba/assets/141175829/d88d51c6-8caa-4ee6-a8a4-f038a8bfacae)
# Installation
* `pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117`
* `pip install packaging`
* `pip install timm==0.4.12`
* `pip install pytest chardet yacs termcolor`
* `pip install submitit tensorboardX`
* `pip install triton==2.0.0`
* `pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs`
## Other requirements:
* Linux System
* NVIDIA GPU
* CUDA 12.0+
# The classification performance of MedMamba
Since MedMamba is suitable for most medical images, you can try applying it to advanced tasks (such as ***multi-label classification***, ***medical image segmentation***, and ***medical object detection***). In addition, we are testing MedMamba with different parameter sizes.
![dataset_01](https://github.com/YubiaoYue/MedMamba/assets/141175829/eac1524c-cd39-47ea-859d-ed109ca1d771)
| Dataset|Task|precision|Sensitivity|Specificity|F1-score|Overall Accuracy|AUC|Model Weight|
|:------:|:--------:|:--------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| ***PAD-UFES-20***    | Multi-Class(6)|38.43|36.94|89.90|35.80|58.80|0.8070|Coming soon!|
| ***Cervical-US***    | Multi-Class(4)|82.67|73.83|94.38|76.32|85.62|0.9524|Coming soon!|
| ***Fetal-US***    | Multi-Class(6)|92.15|93.89|98.73|92.97|93.97 |0.9931|Coming soon!|
| ***CPN-Xray***    |Multi-Class(3) |97.21|97.17|98.54|97.19|97.12|0.9953|Coming soon!|
| ***Kvasir***   |Multi-Class(8)|78.74|78.83|96.97|78.59|78.83 |0.9731|Coming soon!|
| ***Otoscopy2024***    | Multi-Class(9)|86.00|84.44|98.59|85.15|89.45|0.9889|Coming soon!|
| ***[BloodMNIST](https://medmnist.com/)***|Multi-Class(8)|98.31   | 98.38   |98.16   |99.75   |98.26   |-|Coming soon!|
|***[DermaMNIST](https://medmnist.com/)***| Multi-Class(7)|72.88|49.77|43.41|92.11|45.94  |-|Coming soon!|
|***[OrganCMNIST](https://medmnist.com/)***| Multi-Class(11)|95.86|94.67|95.33|99.59|94.65  |-|Coming soon!|
# Citation
If you find this repository useful, please consider the following references. We would greatly appreciate it.
```bibtex
@article{yue2024medmamba,
  title={MedMamba: Vision Mamba for Medical Image Classification},
  author={Yue, Yubiao and Li, Zhenzhang},
  journal={arXiv preprint arXiv:2403.03849},
  year={2024}
}
```
# Acknowledgments
We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba), [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) and [VM-UNet](https://github.com/JCruan519/VM-UNet) for their open-source codes.
# Datasets
## Kvasir
The data is collected using endoscopic equipment at Vestre Viken Health Trust (VV) in Norway. The VV consists of 4 hospitals and provides health care to 470.000 people. One of these hospitals (the Bærum Hospital) has a large gastroenterology department from where training data have been collected and will be provided, making the dataset larger in the future. Furthermore, the images are carefully annotated by one or more medical experts from VV and the Cancer Registry of Norway (CRN). The CRN provides new knowledge about cancer through research on cancer. It is part of South-Eastern Norway Regional Health Authority and is organized as an independent institution under Oslo University Hospital Trust. CRN is responsible for the national cancer screening programmes with the goal to prevent cancer death by discovering cancers or pre-cancerous lesions as early as possible.[Kavsir Dataset](https://datasets.simula.no/kvasir/ "Download it") ![imgs_03](https://github.com/YubiaoYue/MedMamba/assets/141175829/b25b3795-7b30-4736-8fb4-f01787158763)

## Cervical lymph node lesion ultrasound images (Cervical-US)
CLNLUS is a private dataset containing 3392 cervical lymph node ultrasound images. Specifically, these images were obtained from 480 patients in the Ultrasound Department of the Second Affiliated Hospital of Guangzhou Medical University. The entire dataset is divided into four categories by clinical experts based on pathological biopsy results: normal lymph nodes (referred to as normal), benign lymph nodes (referred to as benign), malignant primary lymph nodes (referred to as primary), and malignant metastatic lymph nodes (referred to as metastatic). The number of normal, benign, primary and metastatic images are 1217, 601, 236 and 1338 respectively.![imgs_01](https://github.com/YubiaoYue/MedMamba/assets/141175829/ebdb6dc2-e8a4-4613-af72-9dc88dd04f26)

## FETAL_PLANES_DB: Common maternal-fetal ultrasound images (Fetal-US)
A large dataset of routinely acquired maternal-fetal screening ultrasound images collected from two different hospitals by several operators and ultrasound machines. All images were manually labeled by an expert maternal fetal clinician. Images are divided into 6 classes: four of the most widely used fetal anatomical planes (Abdomen, Brain, Femur and Thorax), the mother’s cervix (widely used for prematurity screening) and a general category to include any other less common image plane. Fetal brain images are further categorized into the 3 most common fetal brain planes (Trans-thalamic, Trans-cerebellum, Trans-ventricular) to judge fine grain categorization performance. Based on FETAL's metadata, we categorize it into six categories. The number of images for each category is as follows: Fetal abdomen (711 images), Fetal brain (3092 images), Fetal femur (1040 images), Fetal thorax (1718 images), Maternal cervis (1626 images), and Other (4213 images). [Dataset URL](https://zenodo.org/records/3904280)
![imgs_04](https://github.com/YubiaoYue/MedMamba/assets/141175829/00beb6e2-6fe6-4cc7-b8f4-e6e00e5697f0)

## Covid19-Pneumonia-Normal Chest X-Ray Images (CPN-Xray)
Shastri et al collected a large number of publicly available and domain recognized X-ray images from the Internet, resulting in CPN-CX. The CPN-CX dataset is divided into 3 categories, namely COVID, NORMAL and PNEUMONIA. All images are preprocessed and resized to 256x256 in PNG format. It helps the researcher and medical community to detect and classify COVID19 and Pneumonia from Chest X-Ray Images using Deep Learning [Dataset URL](https://data.mendeley.com/datasets/dvntn9yhd2/1).![imgs_02](https://github.com/YubiaoYue/MedMamba/assets/141175829/996035b3-2dd5-4c01-b3d4-656f2bf52307)

## Large-scale otoscopy dataset (Otoscopy2024)
This dataset is a supplement to previous work. In [previous publications](https://www.nature.com/articles/s41598-021-90345-w), we collected 20542 endoscopic images of ear infections. On this basis, we have added an additional 2039 images from medical institutions. We will name 22581 endoscopic images of the ear as Otoscopy2024. Otoscopy2024 is a large dataset specifically designed for ear disease classification, consisting of 9 categories: Cholestestoma of middle ear(548 images), Chronic suppurative otitis media(4021 images), External auditory cana bleeding (451 images), Impacted cerumen (6058 images), Normal eardrum (4685 images), Otomycosis external (2507 images), Secretory otitis media (2720 images), Tympanic membrane calcification (1152 images), Acute otitis media (439 images).
![imgs_05](https://github.com/YubiaoYue/MedMamba/assets/141175829/1dcc3bd5-2f89-4afc-b487-1eb4086a58de)
