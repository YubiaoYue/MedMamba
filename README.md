# 🐍MedMamba: Vision Mamba for Medical Image Classification🐍
## This is the official code repository for "***MedMamba: Vision Mamba for Medical Image Classification***"[![arXiv](https://img.shields.io/badge/arXiv-2403.03849-brightgreen.svg)](https://arxiv.org/abs/2403.03849) ![medmamba](https://github.com/YubiaoYue/MedMamba/assets/141175829/dcd0d717-0c33-45b6-9536-784bcd704c14)

# 🎇 Good News 🎇
## Now, everyone can download the pre-trained weights of MedMamba, including our private datasets (password: 2024).If you encounter any difficulties during the download process, please do not hesitate to contact me.💖 ##
![figure4](https://github.com/YubiaoYue/MedMamba/assets/141175829/f338f5a0-4a19-4346-bff5-8ff3eca0d6d6)
# 📝Work Summary📝
Since the era of deep learning, convolutional neural networks (CNNs) and vision transformers (ViTs) have been extensively studied and widely used in medical image classification tasks. Unfortunately, CNN's limitations in modeling long-range dependencies result in poor classification performances. In contrast, ViTs are hampered by the quadratic computational complexity of their self-attention mechanism, making them difficult to deploy in real-world settings with limited computational resources. Recent studies have shown that state space models (SSMs) represented by Mamba can effectively model long-range dependencies while maintaining linear computational complexity. **Inspired by it, we proposed MedMamba, the first vision Mamba for generalized medical image classification. Concretely, we introduced a novel hybrid basic block named SS-Conv-SSM, which purely integrates the convolutional layers for extracting local features with the abilities of SSM to capture long-range dependencies, aiming to model medical images from different image modalities efficiently. By employing the grouped convolution strategy and channel-shuffle operation, MedMamba successfully provides fewer model parameters and a lower computational burden for efficient applications without sacrificing accuracy.** To demonstrate the potential of MedMamba, we conducted extensive experiments using 16 datasets containing ten imaging modalities and 411,007 images. Experimental results show that the proposed MedMamba demonstrates competitive performance in classifying various medical images compared with the state-of-the-art methods. **Our work is aims to establish a new baseline for medical image classification and provide valuable insights for developing more powerful SSM-based artificial intelligence algorithms and application systems in the medical field.**
![Medmamba_net_new_01(1)](https://github.com/YubiaoYue/MedMamba/assets/141175829/160ba28f-eede-4617-83ac-de87c3844664)
![S6_BLOCK](https://github.com/YubiaoYue/MedMamba/assets/141175829/d88d51c6-8caa-4ee6-a8a4-f038a8bfacae)
# 📌Installation📌
* `pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117`
* `pip install packaging`
* `pip install timm==0.4.12`
* `pip install pytest chardet yacs termcolor`
* `pip install submitit tensorboardX`
* `pip install triton==2.0.0`
* `pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl`
* `pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs`
## 📜Other requirements📜:
* Linux System
* NVIDIA GPU
* CUDA 12.0+
# 🔥The classification performance of MedMamba🔥
Since MedMamba is suitable for most medical images, you can try applying it to advanced tasks (such as ***multi-label classification***, ***medical image segmentation***, and ***medical object detection***). In addition, we are testing MedMamba with different parameter sizes.
![dataset-new](https://github.com/YubiaoYue/MedMamba/assets/141175829/547fec48-5572-4ed7-89ee-94dda2a4ca9b)
| Dataset|Task|Precision|Sensitivity|Specificity|F1-score|Overall Accuracy|AUC|Model Weight|
|:------:|:--------:|:--------:|:----------:|:----------:|:----------:|:----------:|:----------:|:----------:|
| **[PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1)**    | Multi-Class(6)|38.4|36.9|89.9|35.8|58.8|0.808|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
| **Cervical-US**    | Multi-Class(4)|81.2|76.2|94.9|78.0|86.2|0.952|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
| **[Fetal-Planes-DB](https://zenodo.org/records/3904280)**    | Multi-Class(6)|92.2|93.9|98.7|93.0|94.0|0.993|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
| **[CPN X-ray](https://data.mendeley.com/datasets/dvntn9yhd2/1)**    |Multi-Class(3) |97.2|97.2|98.5|97.2|97.1|0.995|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
| **[Kvasir](https://datasets.simula.no/kvasir/)**   |Multi-Class(8)|78.7|78.8|97.0|78.6|78.8|0.973|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
| **[Otoscopy2024](https://www.nature.com/articles/s41598-021-90345-w)**| Multi-Class(9)|86.0|84.4|98.6|85.2|89.5|0.989|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
| **[PathMNIST](https://medmnist.com/)**|Multi-Class(9)|94.0|94.7|99.4|94.2|95.3|0.997|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
|**[DermaMNIST](https://medmnist.com/)**| Multi-Class(7)|67.3|50.1|93.6|51.6|77.9|0.917|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
|**[OCTMNIST](https://medmnist.com/)**| Multi-Class(4)|92.8|91.8|97.3|91.8|91.8|0.992|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
|**[PneumoniaMNIST](https://medmnist.com/)**| Multi-Class(2)|92.1|87.0|87.0|88.6|89.9|0.965|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
|**[RetinaMNIST](https://medmnist.com/)**| Multi-Class(5)|35.9|37.7|87.5|36.1|54.3|0.747|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
|**[BreastMNIST](https://medmnist.com/)**| Multi-Class(2)|91.6|72.6|72.6|76.6|85.3|0.825|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
|**[BloodMNIST](https://medmnist.com/)**| Multi-Class(8)|97.7|97.7|99.7|97.7|97.8|0.999|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
|**[OrganAMNIST](https://medmnist.com/)**| Multi-Class(11)|94.4|93.3|99.5|93.8|94.6|0.998|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
|**[OrganCMNIST](https://medmnist.com/)**| Multi-Class(11)|92.2|91.6|99.3|91.7|92.7|0.997|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
|**[OrganSMNIST](https://medmnist.com/)**| Multi-Class(11)|78.0|77.4|98.2|76.3|81.9|0.982|[weights](https://pan.baidu.com/s/1N7AmeyTyh4FKqke7IapkUg )|
# 💞Citation💞
If you find this repository useful, please consider the following references. We would greatly appreciate it.
```bibtex
@article{yue2024medmamba,
  title={MedMamba: Vision Mamba for Medical Image Classification},
  author={Yue, Yubiao and Li, Zhenzhang},
  journal={arXiv preprint arXiv:2403.03849},
  year={2024}
}
```
# ✨Acknowledgments✨
We thank the authors of [VMamba](https://github.com/MzeroMiko/VMamba), [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet) and [VM-UNet](https://github.com/JCruan519/VM-UNet) for their open-source codes.
# 📊Datasets📊
## Kvasir
The data is collected using endoscopic equipment at Vestre Viken Health Trust (VV) in Norway. The VV consists of 4 hospitals and provides health care to 470.000 people. One of these hospitals (the Bærum Hospital) has a large gastroenterology department from where training data have been collected and will be provided, making the dataset larger in the future. Furthermore, the images are carefully annotated by one or more medical experts from VV and the Cancer Registry of Norway (CRN). The CRN provides new knowledge about cancer through research on cancer. It is part of South-Eastern Norway Regional Health Authority and is organized as an independent institution under Oslo University Hospital Trust. CRN is responsible for the national cancer screening programmes with the goal to prevent cancer death by discovering cancers or pre-cancerous lesions as early as possible.[Kavsir Dataset](https://datasets.simula.no/kvasir/ "Download it") ![imgs_03](https://github.com/YubiaoYue/MedMamba/assets/141175829/b25b3795-7b30-4736-8fb4-f01787158763)

## Cervical lymph node lesion ultrasound images (Cervical-US)
CLNLUS is a private dataset containing 3392 cervical lymph node ultrasound images. Specifically, these images were obtained from 480 patients in the Ultrasound Department of the Second Affiliated Hospital of Guangzhou Medical University. The entire dataset is divided into four categories by clinical experts based on pathological biopsy results: normal lymph nodes (referred to as normal), benign lymph nodes (referred to as benign), malignant primary lymph nodes (referred to as primary), and malignant metastatic lymph nodes (referred to as metastatic). The number of normal, benign, primary and metastatic images are 1217, 601, 236 and 1338 respectively.![imgs_01](https://github.com/YubiaoYue/MedMamba/assets/141175829/ebdb6dc2-e8a4-4613-af72-9dc88dd04f26)

## FETAL_PLANES_DB: Common maternal-fetal ultrasound images (Fetal-Planes-DB)
A large dataset of routinely acquired maternal-fetal screening ultrasound images collected from two different hospitals by several operators and ultrasound machines. All images were manually labeled by an expert maternal fetal clinician. Images are divided into 6 classes: four of the most widely used fetal anatomical planes (Abdomen, Brain, Femur and Thorax), the mother’s cervix (widely used for prematurity screening) and a general category to include any other less common image plane. Fetal brain images are further categorized into the 3 most common fetal brain planes (Trans-thalamic, Trans-cerebellum, Trans-ventricular) to judge fine grain categorization performance. Based on FETAL's metadata, we categorize it into six categories. The number of images for each category is as follows: Fetal abdomen (711 images), Fetal brain (3092 images), Fetal femur (1040 images), Fetal thorax (1718 images), Maternal cervis (1626 images), and Other (4213 images). [Dataset URL](https://zenodo.org/records/3904280)
![imgs_04](https://github.com/YubiaoYue/MedMamba/assets/141175829/00beb6e2-6fe6-4cc7-b8f4-e6e00e5697f0)

## Covid19-Pneumonia-Normal Chest X-Ray Images (CPN X-ray)
Shastri et al collected a large number of publicly available and domain recognized X-ray images from the Internet, resulting in CPN-CX. The CPN-CX dataset is divided into 3 categories, namely COVID, NORMAL and PNEUMONIA. All images are preprocessed and resized to 256x256 in PNG format. It helps the researcher and medical community to detect and classify COVID19 and Pneumonia from Chest X-Ray Images using Deep Learning [Dataset URL](https://data.mendeley.com/datasets/dvntn9yhd2/1).![imgs_02](https://github.com/YubiaoYue/MedMamba/assets/141175829/996035b3-2dd5-4c01-b3d4-656f2bf52307)

## Large-scale otoscopy dataset (Otoscopy2024)
This dataset is a supplement to previous work. In [previous publications](https://www.nature.com/articles/s41598-021-90345-w), we collected 20542 endoscopic images of ear infections. On this basis, we have added an additional 2039 images from medical institutions. We will name 22581 endoscopic images of the ear as Otoscopy2024. Otoscopy2024 is a large dataset specifically designed for ear disease classification, consisting of 9 categories: Cholestestoma of middle ear(548 images), Chronic suppurative otitis media(4021 images), External auditory cana bleeding (451 images), Impacted cerumen (6058 images), Normal eardrum (4685 images), Otomycosis external (2507 images), Secretory otitis media (2720 images), Tympanic membrane calcification (1152 images), Acute otitis media (439 images).
![imgs_05](https://github.com/YubiaoYue/MedMamba/assets/141175829/1dcc3bd5-2f89-4afc-b487-1eb4086a58de)
