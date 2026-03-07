# PEHT: Parameter-Efficient Hybrid Transformer for Network Traffic Prediction

This repository contains the implementation and experiments for the paper:

**"Parameter Efficient Hybrid Transformer (PEHT) for Network Traffic Prediction via Dynamic Urban Congestion Integration."**

The proposed **PEHT framework** integrates **urban mobility dynamics with cellular network traffic prediction** using a **parameter-efficient Transformer architecture enhanced with Low-Rank Adaptation (LoRA)**. :contentReference[oaicite:0]{index=0}

---

# Overview

Accurate network traffic prediction is essential for **efficient resource allocation in cellular and wireless infrastructures**, especially in dynamic urban environments where mobility and congestion influence communication demand. :contentReference[oaicite:1]{index=1}

Traditional prediction models often struggle to incorporate **external mobility factors and urban traffic conditions**. The PEHT architecture addresses this challenge by integrating:

- network communication features
- urban mobility and congestion features
- multimodal temporal information

within a **customized Transformer-based architecture**. :contentReference[oaicite:2]{index=2}

---

# Key Contributions

The main contributions of the PEHT framework include:

- **Parameter-Efficient Transformer Architecture**  
  A customized Transformer encoder enhanced with **Low-Rank Adaptation (LoRA)** to reduce the number of trainable parameters.

- **Multimodal Data Fusion**  
  Integration of **urban traffic and congestion features** into the prediction pipeline.

- **Mobility-Aware Traffic Prediction**  
  Explicit modeling of **urban mobility dynamics** to improve network traffic forecasting.

- **Improved Prediction Performance**  
  PEHT outperforms several state-of-the-art models on the Telecom Italia Milan dataset in terms of **RMSE, MAE, and R² metrics**. :contentReference[oaicite:3]{index=3}

---

# PEHT Architecture

The PEHT model consists of three main components:

### 1. Spatio-Temporal Grid Clustering

Raw cellular traffic data is aggregated using a **Grid-to-Virtual-Cell mapping strategy**, where adjacent spatial grids are grouped into **Virtual Base Stations (VBS)** to simulate cellular tower coverage.

### 2. Multimodal Feature Engineering

Multiple data modalities are integrated, including:

- Historical network traffic
- Temporal features
- Static network features
- Dynamic mobility features

These inputs are embedded and fused into a unified representation for the Transformer model.

### 3. Parameter-Efficient Transformer

The prediction core is a **customized Encoder–Decoder Transformer** with:

- Multi-head attention
- Positional encoding
- LoRA-based parameter reduction
- Multimodal feature fusion

The encoder processes historical traffic patterns while the decoder predicts future network demand.

---

# LoRA-Based Parameter Reduction

To handle high-dimensional traffic data efficiently, PEHT integrates **Low-Rank Adaptation (LoRA)** into the Transformer encoder.

Instead of training a full weight matrix:

W ∈ R^(d_out × d_in)


the update is approximated using two low-rank matrices:


ΔW ≈ A B


where:


A ∈ R^(d_out × r)
B ∈ R^(r × d_in)


This significantly reduces trainable parameters.

Example:

| Rank r | Trainable Parameters |
|------|----------------|
| r = 4 | ~34,976 |
| r = 8 | ~69,952 |
| r = 16 | ~139,904 |

Compared to **~19.1 million parameters in the full matrix**, this provides massive efficiency gains. :contentReference[oaicite:4]{index=4}

---

# Experimental Results

Experiments were conducted using the **Telecom Italia Milan dataset** along with several synthetic congestion scenarios.

PEHT consistently outperformed baseline models such as:

- HGCRN
- ST-DenseNet
- STCNet
- ST-Tran
- MVSTGN
- ST2T

Example results:

| Model | RMSE (SMS) | MAE | R² |
|------|-----------|-----|-----|
| ST2T | 21.58 | 14.45 | 0.935 |
| **PEHT (Ours)** | **18.42** | **12.15** | **0.948** |

PEHT also improved prediction accuracy for **Call traffic and Internet traffic**. :contentReference[oaicite:5]{index=5}

---

# Synthetic Scenario Evaluation

Additional experiments were conducted under five simulated conditions:

- Base-station congestion
- Cell-edge congestion
- Vehicular-only mobility
- Pedestrian-only mobility
- Regular traffic patterns

Across all scenarios, the **full PEHT model consistently achieved the best performance** compared to ablated versions without LoRA or feature fusion. :contentReference[oaicite:6]{index=6}



