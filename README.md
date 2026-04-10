# CS542 Final Project: NIH Chest X-Ray Classification

**Names:** Marissa Estramonte, Chaz Gillette, and Anna Harner  
**Student IDs:** 2405291, 2403643, 2397569  
**Date:** 4/07/2026

## GitHub Repository

https://github.com/AnnaMHarner/Deep-Learning-CV-Final-Project

## Accessing the Data

The dataset used in this project is the NIH Chest X-ray dataset, publicly available on Kaggle:

https://www.kaggle.com/datasets/nih-chest-xrays/data/data

To use this dataset locally, download it from the link above and place the contents in a folder named `Chest_XRay+Data/` in the root of the repository. The notebook expects the following files at minimum:

- `Chest_XRay+Data/Data_Entry_2017.csv`
- `Chest_XRay+Data/BBox_List_2017.csv`

## Resources Used

- Rajpurkar, P., et al. (2017). _CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning._ arXiv:1711.05225. https://arxiv.org/abs/1711.05225
- Domain expert interview with a practicing Physician Assistant (PA) conducted as part of the assignment requirements.

## Dataset Credit

Wang et al. (2017). _ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks._ CVPR 2017.

## Contributions

Anna Harner: problem formulation, deep learning justification, data pipeline, metadata preprocessing, dataset class, run_epoch training loop, scenario (b).  
Marissa Estramonte: transfer learning (get_transfer_model), build_optimizer with differential learning rates, evaluate_metrics, result visualizations, scenarios (c) and (d).  
Chaz Gillette: custom ChestXRayCNN architecture, GradCAM, best/worst prediction visualizations, scenario (a), bug fixes for scenarios (c) and (d).
