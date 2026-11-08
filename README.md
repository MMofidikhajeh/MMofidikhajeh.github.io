# Mohammad Mofidikhajeh
**AI Research Engineer | Deep Learning & PyTorch Specialist**  
*Bridging the gap between mathematical theory and optimized, reproducible PyTorch code.*

[GitHub](https://github.com/MMofidikhajeh) | [LinkedIn](https://linkedin.com/in/mohammad-mofidikhajeh) | [Email](mailto:mohammad.mofidi.k@gmail.com) |

---

## 🛠️ Research Engineering Services (Freelance & Consulting)
**You focus on the theory. I will build the PyTorch.**

Many brilliant AI research projects stall at the implementation phase. Translating novel mathematical formulations into optimized, hardware-aware code is a distinct skill set. I partner with PIs, Postdocs, and PhD candidates who have strong theoretical foundations but need an elite engineer to execute. 

**How I can help your lab:**
*   **Paper-to-Code Implementation:** Rapid, bug-free translation of SOTA architectures from arXiv, NeurIPS, or ICLR into modular PyTorch code.
*   **Custom Architecture Engineering:** Modifying core mechanisms (e.g., custom attention logit biases, graph-based routing, novel loss functions) without breaking `FlashAttention` or Distributed Data Parallel (DDP) training.
*   **Reproducible Pipelines:** Building robust, version-controlled training frameworks (WandB, Hydra, Docker) so your experiments are verifiable and easily ablated by reviewers.
*   **Debugging & Optimization:** Profiling memory bottlenecks, fixing vanishing gradients, and ensuring your models actually train efficiently on modern GPUs.

---

## 👤 About Me
I am a Master's researcher in Computer Science (Data Mining) at Tarbiat Modares University, graduating in December 2026, with a unique background that bridges clinical sciences and rigorous machine learning theory. 

My Bachelor's degree is in Radiotherapy, where I managed longitudinal, multivariate patient cohort data and conducted survival analyses. That experience with messy, real-world biological data drove me into Computer Science to build better predictive models. Today, I specialize in **Transformer architectures, representation learning, and graph-theoretic inductive biases**. 

I don't just run existing repositories; I build architectures from scratch, derive the math, and write the dense, highly-parallelized tensor operations required to make them run efficiently on modern hardware. Whether I am applying for a PhD program or consulting for a research lab, my goal is the same: to build AI systems that are mathematically sound, structurally interpretable, and engineering-grade.

---

## 🚀 Featured Research & Engineering Projects

### 1. Semantic Vision Graphormer (SVG) | *Master's Thesis*
**The Theory:** Standard ViTs treat images as unstructured 1D sequences, forcing the model to implicitly learn 2D topology. I modeled images natively as a dual-graph system (a fixed lattice graph for spatial topology, and a dynamic contextual graph for semantic similarity) and injected these biases directly into the self-attention logits.
**The Engineering:** Implemented entirely from scratch in PyTorch. By keeping operations dense and compatible with `FlashAttention`, SVG adds **<1k parameters** and achieves **zero wall-clock latency overhead** while yielding a **4.01% accuracy gain** over ViT baselines. Discovered a layer-wise topological phase transition from texture grouping to semantic assembly.
*   **First-author manuscript in preparation.**
*   [🔗 View Code & Architecture](https://github.com/MMofidikhajeh/SVG)

### 2. Label-Efficient Segmentation via DINOv2 Knowledge Distillation
**The Theory:** Investigating how to transfer dense, self-supervised representations from massive foundation models into lightweight architectures for low-label regimes (e.g., medical imaging).
**The Engineering:** Built a semi-supervised training framework distilling a frozen DINOv2 teacher into a ConvNeXt-Tiny U-Net student. Engineered controlled ablation pipelines across stratified partial-label budgets to evaluate IoU, Dice, and boundary sensitivity.
*   [🔗 View Repository](https://github.com/MMofidikhajeh/label-efficient-segmentation)

### 3. Metric Learning for Out-of-Distribution (OOD) Detection
**The Theory:** Shaping embedding geometries to reliably detect distribution shifts, anomalous samples, and unreliable predictions—critical for safe deployment in clinical or industrial settings.
**The Engineering:** Developed a modular PyTorch evaluation framework benchmarking 5 scoring methods (MSP, Energy, Mahalanobis, kNN, prototype distance) against 4 representation learning objectives (Cross-Entropy, Triplet, Supervised Contrastive, Synthetic-Outlier). Integrated UMAP for embedding geometry visualization.
*   [🔗 View Repository](https://github.com/MMofidikhajeh/ood-detection-metric-learning)

### 4. SaaS Customer Churn Analytics & Survival Modeling
**The Theory:** Applying time-to-event modeling and predictive analytics to multivariate longitudinal data to understand subscriber retention and concept drift.
**The Engineering:** End-to-end data pipeline handling messy, real-world tabular data, feature engineering, and survival analysis, demonstrating adaptability beyond computer vision into time-series and business logic.
*   [🔗 View Repository](https://github.com/MMofidikhajeh/saas-customer-churn-analytics)

### 5. Superstore Sales & Performance Analysis
**The Theory:** Statistical analysis and data mining for actionable business intelligence.
**The Engineering:** Clean data wrangling, exploratory data analysis (EDA), and visualization pipelines using Pandas and Seaborn.
*   [🔗 View Repository](https://github.com/MMofidikhajeh/Superstore_Analysis)

---

## 🎓 Teaching & Technical Communication
**Invited Technical Instructor | "Hands-On Large Language Models" Graduate Study Group**
*Tarbiat Modares University | September 2026*
Selected by the department head to lead the most mathematically rigorous sessions of a 4-week graduate reading group. 
*   Synthesized complex mathematical concepts (self-attention mechanics, positional encodings, KV-caching) into accessible, first-principles frameworks.
*   Conducted live PyTorch code walkthroughs and debugging sessions, bridging the gap between theoretical papers and practical implementation for my peers.

---

## 📬 Let's Collaborate
**For PhD Admissions:** I am actively seeking PhD positions in Europe starting in early 2027, focusing on robust ML, representation learning, and clinical/spatio-temporal AI.
**For Labs & Researchers:** I am available for part-time, freelance research engineering. If you have a mathematical formulation that needs to be turned into a robust PyTorch pipeline, let's talk.

📧 **Email:** [mohammad.mofidi.k@gmail.com](mailto:mohammad.mofidi.k@gmail.com)  
💼 **LinkedIn:** [linkedin.com/in/mohammad-mofidikhajeh](https://linkedin.com/in/mohammad-mofidikhajeh)  
💻 **GitHub:** [github.com/MMofidikhajeh](https://github.com/MMofidikhajeh)  
🌐 **Website:** [mmofidikhajeh.github.io](https://mmofidikhajeh.github.io/)
