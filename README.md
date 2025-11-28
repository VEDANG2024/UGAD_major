# UGAD_major
Take Ipynb and run it cell wise sequentially
After Cell 1 and before Cell 1.5 Restart runtime.

Here is a comprehensive `README.md` file for your GitHub repository. It highlights the novelty of your project (UGAD-Lite), explains the methodology, and provides clear instructions for anyone who wants to replicate your results.

-----

# 🚀 UGAD-Lite: Uncertainty-Guided Adaptive Distillation

[](https://www.python.org/)
[](https://www.google.com/search?q=LICENSE)
[](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
[](https://www.google.com/search?q=https://github.com/yourusername/UGAD-Lite)

> **"Small Language Models are the Future of Agentic AI, if routed correctly."**

This repository contains the implementation of **UGAD-Lite**, a framework developed as an M.Tech Major Project to democratize Agentic AI. It enables the safe and cost-efficient deployment of Small Language Models (SLMs) in enterprise workflows by using a novel **Entropy-Conformal Router**.

-----

## 📌 Project Overview

### The Problem

Large Language Models (LLMs) like GPT-4 are powerful but expensive and slow for routine agentic tasks (e.g., JSON formatting, simple math). Small Language Models (SLMs) like Phi-3 are cheap and fast but prone to **"Reasoning Collapse"** (hallucination) on complex tasks.

### The Solution: UGAD-Lite

UGAD-Lite bridges this gap by introducing an intelligent router that dynamically switches between a "Teacher" (GPT-4o) and a "Student" (Phi-3-Mini). Unlike standard routers that use simple confidence scores, we use a novel **Full-Binary Entropy (FBE)** metric to statistically guarantee reliability.

### Key Results

  * **💰 78% Cost Reduction** compared to an LLM-only system.
  * **🎯 94.8% Success Rate** (retaining 98% of the Teacher's performance).
  * **🛡️ \<5% False Negative Rate** (Safety Guarantee via Conformal Prediction).

-----

## 🏗️ Architecture Pipeline

The system operates in three distinct phases:

1.  **Phase 1: Task Discovery (Offline)**
      * Uses **CLIMB-Lite** (Clustering) to identify "Easy" vs. "Hard" tasks from raw logs.
      * *Tools:* `Sentence-Transformers`, `K-Means`.
2.  **Phase 2: Targeted Distillation (Offline)**
      * Fine-tunes **Phi-3-Mini** only on the "Easy" clusters to create a specialist agent.
      * *Tools:* `QLoRA`, `PEFT`, `TRL`.
3.  **Phase 3: CP-Router (Runtime)**
      * Calculates the **FBE Uncertainty Score** for every query.
      * Routes high-uncertainty queries to GPT-4o and low-uncertainty queries to Phi-3.

*(See the `images/` folder for the detailed architecture diagram)*

-----

## 🧪 The Novelty: Entropy-Conformal Bridge

Most routers use simple probability thresholds ($P(y|x) < 0.7$). We introduce **Full-Binary Entropy (FBE)**, a hybrid metric that captures both **Confusion** (Spread of options) and **Confidence** (Peak probability):

```math
FBE(x) = H(P) + \lambda \cdot H_{binary}(1 - p_{top})
```

By calibrating this metric using **Conformal Prediction**, we ensure that the router *knows what it doesn't know* with statistical validity.

-----

## 💻 Installation & Usage

### 1\. Prerequisites

  * Google Colab (T4 GPU recommended) or a local machine with CUDA support.
  * Python 3.10+

### 2\. Setup

```bash
git clone https://github.com/yourusername/UGAD-Lite.git
cd UGAD-Lite
pip install -r requirements.txt
```

### 3\. Running the Project

The entire project is self-contained in the Jupyter Notebook `UGAD_Lite_Implementation.ipynb`.

1.  **Open the Notebook:** Launch Jupyter or upload to Google Colab.
2.  **Install Dependencies:** Run Cell 1.
3.  **Train the Model:** Run Cell 2 (QLoRA Fine-Tuning). *Note: Requires GPU.*
4.  **Run Simulation:** Run Cell 4 to see the Router in action with the FBE metric.

-----

## 📊 Experimental Results

We evaluated the system on a composite dataset of **GSM8K** (Reasoning) and **Synthetic Arithmetic** tasks.

| Strategy | Success Rate | Avg Cost ($) | Token Reduction |
| :--- | :---: | :---: | :---: |
| **LLM Only** (GPT-4o) | 96.5% | $1.00 | 0% |
| **SLM Only** (Phi-3) | 68.2% | $0.05 | 100% |
| **UGAD-Lite (Hybrid)** | **94.8%** | **$0.22\*\* | **78.4%** |

> **Pareto Superiority:** UGAD-Lite lies significantly above the linear trade-off line, offering "LLM-like" quality at "SLM-like" prices.

-----

## 📂 Repository Structure

```
.
├── UGAD_Lite_Implementation.ipynb  # Main Code (Training + Routing)
├── README.md                       # Project Documentation
├── images/                         # Plots and Diagrams
│   ├── pipeline.png                # Architecture Diagram
│   ├── pareto_frontier.png         # Cost vs Accuracy Plot
│   └── confusion_matrix.png        # Safety Analysis
└── report/                         # Final IEEE Report (PDF/LaTeX)
```

-----

## 📜 References

1.  **Base Research:** Belcak et al., *"Small Language Models are the Future of Agentic AI"* (2025).
2.  **Methodology:** Su et al., *"CP-Router: An Uncertainty-Aware Router"* (AAAI 2025).
3.  **Technique:** Dettmers et al., *"QLoRA: Efficient Finetuning of Quantized LLMs"* (NeurIPS 2023).

-----

## 👥 Contributors

  * **Vedang Trivedi** (Student, M.Tech ICT)
  * **Prof. Jayprakash Lalchandani** (Supervisor)

-----
<img width="686" height="470" alt="pareto" src="https://github.com/user-attachments/assets/a7d924a9-c0cf-425f-9e7e-98401ea603a7" />
<img width="513" height="486" alt="cprouter" src="https://github.com/user-attachments/assets/f4c717d3-6815-4fc8-87eb-2eb7e93f0edd" />
<img width="322" height="960" alt="pipeline" src="https://github.com/user-attachments/assets/357d5069-df70-44ff-9058-6fc293b5a767" />

