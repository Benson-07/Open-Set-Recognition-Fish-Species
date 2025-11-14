🐟 Open-Set Recognition for Marine Fish Species

CLIP Vision Transformers + Mahalanobis + Deep Mahalanobis

This repository contains a complete Open-Set Recognition (OSR) pipeline for marine fish species using CLIP ViT models and Mahalanobis-based unknown detection.
The goal: build a classifier that not only recognises known fish species, but also rejects unseen/unknown species reliably—crucial for real-world marine monitoring.

📌 Project Highlights

6 known fish species used for training

3 unseen species used only during testing

CLIP ViT-B/32, ViT-B/16, ViT-L/14 as frozen feature extractors

Linear classifier for closed-set classification

MSP, Energy, Mahalanobis, Deep Mahalanobis for OSR

State-of-the-art open-set performance:

ViT-L/14 + Deep Mahalanobis → AUROC = 0.994

Closed-set accuracy (known classes only) → ~99.9%

📂 Dataset

Known Classes

Gilt-Head Bream

Hourse Mackerel

Red Mullet

Red Sea Bream

Striped Red Mullet

Trout

Unknown Classes

Sea Bass

Black Sea Sprat

Shrimp

🧠 Method Overview
1️⃣ Feature Extraction (CLIP)

We use frozen CLIP backbones:

ViT-B/32

ViT-B/16

ViT-L/14

For each image, we extract the L2-normalized CLS embedding.

2️⃣ Closed-Set Classifier

A simple multinomial Logistic Regression is trained on known classes.

Why logistic regression?

CLIP embeddings are extremely powerful

Ensures good interpretability

Avoids overfitting

Provides clean logits for OSR methods

3️⃣ Open-Set Recognition Methods
MSP (Max Softmax Probability)

Simple confidence score → weakest performance.

Energy Score

Uses log-sum-exp of logits.

Mahalanobis Distance

Computes distance to class-mean embedding:

Estimates shared covariance using Ledoit–Wolf

Excellent separation of known vs unknown samples

Deep Mahalanobis

Extracts features from multiple transformer layers:

For ViT-L/14 → blocks [3, 7, 15, 23]

For ViT-B models → blocks [2, 5, 8, 11]

Fuses distances from all layers → strongest OSR performance.


👉 Best Model:
ViT-L/14 + Deep Mahalanobis (AUROC = 0.994)

