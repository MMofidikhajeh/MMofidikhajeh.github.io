# Mohammad Mofidikhajeh
**Machine Learning Researcher & PyTorch Engineer**

I build custom architectures from scratch and turn dense math into highly parallelized tensor operations. Currently finishing my MSc in Computer Science (Data Mining) at Tarbiat Modares University (graduating December 2026).

[GitHub](https://github.com/MMofidikhajeh) | [LinkedIn](https://linkedin.com/in/mohammad-mofidikhajeh) | [Email](mailto:mohammad.mofidi.k@gmail.com) | [Website](https://mmofidikhajeh.github.io/)

---

### Research Engineering & Consulting
Many brilliant papers stall at implementation. Translating novel mathematical formulations into hardware-aware code is a specific headache. I partner with PIs and PhD candidates who have the theory locked down but need someone to execute the PyTorch without breaking `FlashAttention` or Distributed Data Parallel (DDP) training.

**How I can help your lab:**
*   **Paper-to-Code:** Rapid translation of SOTA architectures from arXiv/NeurIPS into modular PyTorch.
*   **Custom Architectures:** Modifying core mechanisms (custom attention logit biases, graph-based routing, novel loss functions) while keeping operations dense.
*   **Reproducible Pipelines:** Building robust training frameworks (WandB, Hydra, Docker) so your experiments are actually verifiable by reviewers.
*   **Debugging:** Profiling memory bottlenecks and fixing vanishing gradients so models train efficiently on modern GPUs.

---

### About Me
I started out in Radiotherapy. Managing longitudinal patient data and running survival analyses showed me exactly how messy real-world biological data is, and how brittle standard statistical models can be. That frustration pushed me into Computer Science.

Today, I specialize in representation learning, Transformer architectures, and graph-theoretic inductive biases. I don't just fine-tune HuggingFace models. I write the dense tensor operations required to make custom architectures actually train efficiently on modern GPUs. Whether I'm working on my Master's thesis or consulting for a lab, I care about two things: mathematical rigor and code that runs at scale.

---

### Featured Research & Engineering

**Semantic Vision Graphormer (SVG) | *Master's Thesis***
Standard ViTs treat images as unstructured 1D sequences, forcing the model to implicitly learn 2D topology. For my thesis, I modeled images natively as a dual-graph system (a fixed lattice for spatial topology, and a dynamic contextual graph for semantic similarity) and injected these biases directly into the self-attention logits.

*The engineering reality:* I built this entirely from scratch in PyTorch. The hard part was keeping operations dense and compatible with `FlashAttention`. The result: SVG adds less than 1k parameters and zero wall-clock latency overhead, while yielding a 4.01% accuracy gain over ViT baselines. I also observed a layer-wise topological phase transition from texture grouping to semantic assembly.
*   First-author manuscript currently in preparation.
*   [🔗 Code & Architecture](https://github.com/MMofidikhajeh/SVG)

**Label-Efficient Segmentation via DINOv2 Knowledge Distillation**
Investigating how to transfer dense, self-supervised representations from massive foundation models into lightweight architectures for low-label regimes (like medical imaging). Built a semi-supervised framework distilling a frozen DINOv2 teacher into a ConvNeXt-Tiny U-Net student. Ran controlled ablations across stratified partial-label budgets to evaluate IoU, Dice, and boundary sensitivity.
*   [🔗 Repository](https://github.com/MMofidikhajeh/label-efficient-segmentation)

**Metric Learning for Out-of-Distribution (OOD) Detection**
Shaping embedding geometries to reliably detect distribution shifts and anomalous samples. I built a modular PyTorch evaluation framework benchmarking five scoring methods (MSP, Energy, Mahalanobis, kNN, prototype distance) against four representation learning objectives (CE, Triplet, SupCon, Synthetic-Outlier). Includes UMAP pipelines for visualizing embedding geometry.
*   [🔗 Repository](https://github.com/MMofidikhajeh/ood-detection-metric-learning)

---

### Other Applied Work
*   **SaaS Customer Churn & Survival Modeling:** Built an end-to-end pipeline for time-to-event modeling on messy, longitudinal tabular data to track subscriber retention and concept drift. [🔗 Code](https://github.com/MMofidikhajeh/saas-customer-churn-analytics)

---

### Teaching
**Graduate Study Group: "Hands-On Large Language Models"**
*Tarbiat Modares University | September 2026*
Ran the mathematically rigorous sessions for a 4-week graduate reading group. I broke down self-attention mechanics, positional encodings, and KV-caching from first principles, and ran live PyTorch code walkthroughs to show my peers how the math actually maps to the tensors.

---

### Let's Collaborate
*   **PhD Admissions:** I am actively seeking PhD positions in Europe starting in early 2027, focusing on robust ML, representation learning, and clinical/spatio-temporal AI.
*   **Labs & Researchers:** I am available for part-time, freelance research engineering. If you have a mathematical formulation that needs to be turned into a robust PyTorch pipeline, let's talk.

📧 [mohammad.mofidi.k@gmail.com](mailto:mohammad.mofidi.k@gmail.com)
💼 [linkedin.com/in/mohammad-mofidikhajeh](https://linkedin.com/in/mohammad-mofidikhajeh)
💻 [github.com/MMofidikhajeh](https://github.com/MMofidikhajeh)
