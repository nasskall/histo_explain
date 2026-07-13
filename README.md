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

Uploads whose longest side exceeds 512 px are downscaled to that limit before inference, preserving aspect ratio, and the app reports when it has done so. The VGG16 backbone is fully convolutional, so activation memory grows with image area: peak usage is roughly 0.7 GB at 224 px, 1.4 GB at 512 px and 3.9 GB at 1024 px, against about 2.7 GB on Streamlit Community Cloud. Without the cap a large patch exhausts the container's memory and the app is terminated. Images already below the limit are used at their native resolution.

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

The 512 px cap (`MAX_SIDE` in `main.py`) is set by the memory available on Streamlit Community Cloud, not by the model, which accepts any input size. Running on a host with more memory allows the limit to be raised or removed.
