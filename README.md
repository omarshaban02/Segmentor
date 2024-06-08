# Segmentor
## Description
Segmentor is a desktop application built with PyQt to perform various thresholding and segmentation techniques on grayscale and color images. The application implements optimal thresholding, Otsu's method, spectral thresholding, and local thresholding for grayscale images. For grayscale and color images, it supports unsupervised segmentation using k-means, region growing, agglomerative clustering, and mean shift methods.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Features](#features)
4. [Contributors](#Contributors)

## Installation
To install the project, clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/Zoz-HF/Segmentor.git

# Navigate to the project directory
cd Segmentor
```

## Usage
To run the application, use the following command:

```bash
python index.py
```

## Features
### Thresholding for Grayscale Images
- Optimal Thresholding
- Otsu's Method
- Spectral Thresholding (more than 2 modes)
- Local Thresholding

### Unsupervised Segmentation for Grayscale and Color Images
- K-means Clustering
  ![K-means](assets/km.png)
- Region Growing
  ![RG](assets/rg1.png)
- Agglomerative Clustering
  ![AC](assets/agl.png)
- Mean Shift Method
  ![MS](assets/ms.png)

## Contributors

- [Ziyad El Fayoumy](https://github.com/Zoz-HF)
- [Assem Hussein](https://github.com/RushingBlast)
- [Mohamed Sayed Diab](https://github.com/MohamedSayedDiab)
- [Abdel Rahman Shawky](https://github.com/AbdulrahmanGhitani)
- [Omar Shaban](https://github.com/omarshaban02)
