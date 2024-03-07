# MedMamba: Vision Mamba for Medical Image Classification
This is the official code repository for "MedMamba: Vision Mamba for Medical Image Classification". [Arxiv Paper](https://arxiv.org/abs/2403.03849)
![logo](https://github.com/YubiaoYue/MedMamba/assets/141175829/f55b3a61-26ab-4256-8fa9-a8f0022c63a2)
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
# Pre-trained weights
Coming soon.
# Acknowledgments
We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba), [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) and [VM-UNet](https://github.com/JCruan519/VM-UNet) for their open-source codes.
# Citation
If you find this repository useful, please consider the following references. We would greatly appreciate it.
```bibtex
@misc{yue2024medmamba,
      title={MedMamba: Vision Mamba for Medical Image Classification}, 
      author={Yubiao Yue and Zhenzhang Li},
      year={2024},
      eprint={2403.03849},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
# Datasets
## Kavsir
The data is collected using endoscopic equipment at Vestre Viken Health Trust (VV) in Norway. The VV consists of 4 hospitals and provides health care to 470.000 people. One of these hospitals (the Bærum Hospital) has a large gastroenterology department from where training data have been collected and will be provided, making the dataset larger in the future. Furthermore, the images are carefully annotated by one or more medical experts from VV and the Cancer Registry of Norway (CRN). The CRN provides new knowledge about cancer through research on cancer. It is part of South-Eastern Norway Regional Health Authority and is organized as an independent institution under Oslo University Hospital Trust. CRN is responsible for the national cancer screening programmes with the goal to prevent cancer death by discovering cancers or pre-cancerous lesions as early as possible.[Kavsir Dataset](https://datasets.simula.no/kvasir/ "Download it") ![imgs_03](https://github.com/YubiaoYue/MedMamba/assets/141175829/b25b3795-7b30-4736-8fb4-f01787158763)

## Cervical lymph node lesion ultrasound images (CLNLUS)
CLNLUS is a private dataset containing 3392 cervical lymph node ultrasound images. Specifically, these images were obtained from 480 patients in the Ultrasound Department of the Second Affiliated Hospital of Guangzhou Medical University. The entire dataset is divided into four categories by clinical experts based on pathological biopsy results: normal lymph nodes (referred to as normal), benign lymph nodes (referred to as benign), malignant primary lymph nodes (referred to as primary), and malignant metastatic lymph nodes (referred to as metastatic). The number of normal, benign, primary and metastatic images are 1217, 601, 236 and 1338 respectively.![imgs_01](https://github.com/YubiaoYue/MedMamba/assets/141175829/ebdb6dc2-e8a4-4613-af72-9dc88dd04f26)

## FETAL_PLANES_DB: Common maternal-fetal ultrasound images
A large dataset of routinely acquired maternal-fetal screening ultrasound images collected from two different hospitals by several operators and ultrasound machines. All images were manually labeled by an expert maternal fetal clinician. Images are divided into 6 classes: four of the most widely used fetal anatomical planes (Abdomen, Brain, Femur and Thorax), the mother’s cervix (widely used for prematurity screening) and a general category to include any other less common image plane. Fetal brain images are further categorized into the 3 most common fetal brain planes (Trans-thalamic, Trans-cerebellum, Trans-ventricular) to judge fine grain categorization performance. Based on FETAL's metadata, we categorize it into six categories. The number of images for each category is as follows: Fetal abdomen (711 images), Fetal brain (3092 images), Fetal femur (1040 images), Fetal thorax (1718 images), Maternal cervis (1626 images), and Other (4213 images). [Dataset URL](https://zenodo.org/records/3904280)
![imgs_04](https://github.com/YubiaoYue/MedMamba/assets/141175829/00beb6e2-6fe6-4cc7-b8f4-e6e00e5697f0)

## Covid19-Pneumonia-Normal Chest X-Ray Images
Shastri et al collected a large number of publicly available and domain recognized X-ray images from the Internet, resulting in CPN-CX. The CPN-CX dataset is divided into 3 categories, namely COVID, NORMAL and PNEUMONIA. All images are preprocessed and resized to 256x256 in PNG format. It helps the researcher and medical community to detect and classify COVID19 and Pneumonia from Chest X-Ray Images using Deep Learning [Dataset URL](https://data.mendeley.com/datasets/dvntn9yhd2/1).![imgs_02](https://github.com/YubiaoYue/MedMamba/assets/141175829/996035b3-2dd5-4c01-b3d4-656f2bf52307)

## Large-scale otoscopy dataset
This dataset is a supplement to previous work. In [previous publications](https://www.nature.com/articles/s41598-021-90345-w), we collected 20542 endoscopic images of ear infections. On this basis, we have added an additional 2039 images from medical institutions. We will name 22581 endoscopic images of the ear as Otoscopy2024. Otoscopy2024 is a large dataset specifically designed for ear disease classification, consisting of 9 categories: Cholestestoma of middle ear(548 images), Chronic suppurative otitis media(4021 images), External auditory cana bleeding (451 images), Impacted cerumen (6058 images), Normal eardrum (4685 images), Otomycosis external (2507 images), Secretory otitis media (2720 images), Tympanic membrane calcification (1152 images), Acute otitis media (439 images).
![imgs_05](https://github.com/YubiaoYue/MedMamba/assets/141175829/1dcc3bd5-2f89-4afc-b487-1eb4086a58de)
