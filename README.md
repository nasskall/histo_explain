# histo_explain

[![Live demo](https://img.shields.io/badge/Live%20demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://nasskall-histo-explain-main-iu96ys.streamlit.app/)

**Live demo:** https://nasskall-histo-explain-main-iu96ys.streamlit.app/

Interactive explainability demo for deep-learning classification of **histopathology images**. The app classifies an uploaded histopathology patch as **benign** or **malignant** with a VGG16 model and visualizes *why* the model reached its decision using several complementary attribution methods.

## What it does

Upload a histopathology image and the app returns the predicted class and probability, alongside four visual explanations:

- **Grad-CAM**: class-activation heatmap over the convolutional features.
- **Guided Grad-CAM**: high-resolution saliency combining guided backpropagation with Grad-CAM.
- **Important regions**: contours of the most influential areas extracted from the activation map.
- **Superpixel importance map**: Grad-CAM importance aggregated over superpixels, with a choice of segmentation algorithm (Felzenszwalb, SLIC, or Quickshift) and an adjustable importance threshold and per-algorithm parameters in the sidebar.

The goal is to make the model's reasoning inspectable for pathologists and researchers, in line with faithfulness-oriented interpretability of medical image classifiers.

## Related publications

This demo accompanies research on classification and interpretation of histopathology and microscopy images:

- Kallipolitis, A., Revelos, K., Maglogiannis, I. (2021). Ensembling EfficientNets for the classification and interpretation of histopathology images. *Algorithms*, 14(10), 278.
- Kallipolitis, A., Yfantis, P., Maglogiannis, I. (2023). Improving explainability results of convolutional neural networks in microscopy images. *Neural Computing and Applications*, 35(29), 21535-21553.

## Tech stack

Python, Streamlit, TensorFlow / Keras (VGG16), OpenCV, scikit-image, NumPy, Pillow.

## Run locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

Then open the local URL Streamlit prints (default http://localhost:8501). A trained VGG16 model is expected under `models/vgg16_model`.

## Deployment

The app is deployed on Streamlit Community Cloud from this repository (branch `main`, entry point `main.py`, Python 3.11). Every push to `main` triggers an automatic redeploy.

## Notes

The bundled model is intended for demonstration and research use only. It is not a medical device and must not be used for diagnosis.
