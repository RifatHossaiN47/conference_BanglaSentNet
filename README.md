# 🇧🇩 BanglaSentNet: A Hybrid Deep Learning Framework for Multi-Aspect Sentiment Analysis in Bangla E-Commerce Reviews

<p align="center">
  <a href="https://link.springer.com/chapter/10.1007/978-3-032-11352-8_20"><img src="https://img.shields.io/badge/Springer%20Nature-CCIS%20Vol%202682-005696?style=for-the-badge&logo=springer&logoColor=white" alt="Springer CCIS" /></a>
  <a href="https://doi.org/10.1007/978-3-032-11352-8_20"><img src="https://img.shields.io/badge/DOI-10.1007%2F978--3--032--11352--8__20-blue?style=for-the-badge" alt="DOI" /></a>
  <a href="https://link.springer.com/conference/icdsaia"><img src="https://img.shields.io/badge/Conference-ICDSAIA%202025-green?style=for-the-badge" alt="ICDSAIA 2025" /></a>
  <img src="https://img.shields.io/badge/Weighted%20F1-0.88-success?style=for-the-badge" alt="F1 Score" />
  <img src="https://img.shields.io/badge/Accuracy-85.0%25-orange?style=for-the-badge" alt="Accuracy" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License" /></a>
</p>

---

## ⚡ At a Glance (30-Second Summary)

Online customer reviews in Bengali often express contrasting opinions in the very same sentence — for example, praise for product quality alongside frustration with delivery delay:
> *"কাপড়ের কোয়ালিটি চমৎকার কিন্তু ডেলিভারি পেতে ৭ দিন লাগলো"*  
> ➔ **Quality: Positive** | **Service: Negative**

Standard single-label classifiers fail to disentangle these nuances. **BanglaSentNet** solves this by introducing a **hybrid, dynamic length-adaptive ensemble** combining:
1. **BanglaBERT Transformer** (contextual self-attention for complex semantics)
2. **BiLSTM, LSTM, and GRU** (sequential and temporal pattern learning)
3. **Dual Static + Contextual Embeddings** (GloVe 2.5B tokens + domain FastText + BanglaBERT)
4. **Length-Adaptive Weighting** (dynamically adjusts model confidence based on review length)

Benchmarked on **8,755 human-annotated e-commerce reviews** across four commercial facets (**Quality, Service, Price, Decoration**), BanglaSentNet achieves a **0.88 weighted F1-score** and **85.0% classification accuracy**, outperforming standalone deep learning and classical machine learning models.

---

## 📚 Publication Details

* **Conference:** International Conference on Data Science, AI and Applications (**ICDSAIA 2025**)
* **Book Series:** *Communications in Computer and Information Science (CCIS)*, Volume 2682, pp. 283–300
* **Publisher:** Springer Nature Switzerland, Cham
* **Online Publication:** 02 January 2026
* **Print ISBN:** `978-3-032-11351-1` | **Online ISBN:** `978-3-032-11352-8`
* **Direct DOI:** [https://doi.org/10.1007/978-3-032-11352-8_20](https://doi.org/10.1007/978-3-032-11352-8_20)
* **Springer Chapter Link:** [Read on SpringerLink](https://link.springer.com/chapter/10.1007/978-3-032-11352-8_20)

---

## 🏛️ Framework Architecture

<p align="center">
  <img src="figures/banglasentnet_architecture.png" alt="Proposed BanglaSentNet Hybrid Ensemble Architecture" width="820px" />
</p>

### How BanglaSentNet Works
1. **Two-Stage Preprocessing Pipeline:** Cleans noise, strips emojis/HTML, normalizes Unicode variations, and standardizes spellings via the *Bangla Academy Accessible Dictionary*.
2. **Dual-Embedding Fusion:**
   $$\mathbf{E}_{\text{final}} = \alpha \cdot \mathbf{E}_{\text{static}} + \beta \cdot \mathbf{E}_{\text{contextual}}$$
   Combines 300-d pre-trained **GloVe** (2.5B Bangla tokens) and e-commerce **FastText** with 768-d **BanglaBERT** contextual states.
3. **Orthogonal Neural Classifiers (Table 4 in Paper):**
   * **BanglaBERT:** 12-layer transformer, 768 hidden units, 12 attention heads.
   * **BiLSTM:** 2 layers, $128 \times 2$ units (forward & backward context tracking).
   * **LSTM:** 2 layers, 256 units (progressive sentiment sequence modeling).
   * **GRU:** 2 layers, 200 units (efficient local feature gating).
4. **Dynamic Weighted Ensemble Voting:** Model weights adapt dynamically according to the review length.
5. **Calibrated Decision Boundary:** Aspect classification calibrated at $\tau = 0.5$ for high precision and recall balance.

---

## 📈 Benchmark & Experimental Results

Evaluated using a **70% train / 15% validation / 15% test** split with macro-averaged metrics on multi-label evaluation.

### 🏆 Final Model Comparison (Table 6 in Paper)

| Model Architecture | Accuracy | Precision | Recall | **F1-Score** | Key Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LSTM Baseline** | 71.0% | 0.82 | 0.77 | 0.80 | Sequential forward dependency tracking |
| **BiLSTM Baseline** | 73.0% | 0.84 | 0.79 | 0.82 | Bidirectional context in free-order Bangla |
| **GRU Baseline** | 74.0% | 0.83 | 0.78 | 0.76 | Fast gating for local sentiment patterns |
| **BanglaBERT Alone** | 78.0% | 0.87 | 0.81 | 0.85 | Deep contextual self-attention |
| **BanglaSentNet (Proposed)** | **85.0%** | **0.90** | **0.86** | **0.88** | **Best overall (+3.0% F1, +7.0% Accuracy)** |

### 🔬 Machine Learning & Deep Learning Baselines (Table 5 in Paper)

| Category | Model & Feature Configuration | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Classical ML** | Logistic Regression (LR) + TF-IDF | 40.0% | 0.77 | 0.37 | 0.47 |
| | Support Vector Machine (SVM) + TF-IDF | 49.0% | 0.80 | 0.48 | 0.58 |
| | Random Forest (RF) + TF-IDF | 43.0% | 0.75 | 0.42 | 0.50 |
| **Deep Learning** | CNN + GloVe Embeddings | 59.0% | 0.78 | 0.73 | 0.75 |
| | LSTM + FastText Embeddings | 66.0% | 0.77 | 0.64 | 0.70 |
| | BiLSTM + Keras Embeddings | 56.0% | 0.80 | 0.75 | 0.77 |
| | GRU + GloVe Embeddings | 64.0% | 0.80 | 0.75 | 0.77 |

### 🧩 Ablation Study: Impact of Components (Table 7 in Paper)

| Configuration | Accuracy | Precision | Recall | F1-Score | Degradation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Full BanglaSentNet** | **0.85** | **0.90** | **0.86** | **0.88** | *Baseline* |
| Without BanglaBERT | 0.72 | 0.78 | 0.73 | 0.75 | **-13.0% F1** (Largest drop) |
| Without BiLSTM | 0.76 | 0.81 | 0.77 | 0.79 | **-9.0% F1** |
| Without LSTM | 0.78 | 0.83 | 0.79 | 0.81 | **-7.0% F1** |
| Without GRU | 0.75 | 0.80 | 0.76 | 0.78 | **-10.0% F1** |

---

## ⚖️ Dynamic Ensemble Weight Allocation (Table 8 in Paper)

Review length significantly impacts which neural architecture performs best. BanglaSentNet automatically shifts weights based on review length:

| Review Length Group | Word Count | BanglaBERT | BiLSTM | LSTM | GRU | Analytical Insight |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Short Reviews** | $< 10$ words | 0.25 | **0.30** | 0.25 | 0.20 | BiLSTM & LSTM excel on concise, punchy phrases |
| **Medium Reviews** | $10 - 20$ words | **0.40** | 0.25 | 0.20 | 0.15 | Balanced contextual and sequential voting |
| **Long Reviews** | $> 20$ words | **0.45** | 0.20 | 0.20 | 0.15 | BanglaBERT self-attention resolves complex multi-clause dependencies |

---

## 📦 Dataset Distribution & Aspect Breakdown

Annotated by CS researchers and NLP specialists across leading Bangladeshi e-commerce platforms (**8,755 verified reviews**):

<p align="center">
  <img src="figures/platform_distribution.png" alt="Platform Distribution" width="370px" />
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="figures/confusion_matrix.png" alt="Confusion Matrix on Test Data" width="410px" />
</p>

### Distribution by Aspect Category (Table 2 in Paper)

| Aspect Category | Total Reviews | Positive | Negative | Neutral | Target Customer Focus |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Quality** (কোয়ালিটি / মান) | **4,000** | 2,400 | 1,200 | 400 | Fabric quality, build, authenticity, durability |
| **Price** (দাম / মূল্য) | **2,500** | 1,500 | 900 | 100 | Pricing fairness, value for money, discounts |
| **Decoration** (ডিজাইন / লুক) | **1,255** | 750 | 400 | 105 | Visual aesthetics, color accuracy, packaging |
| **Service** (ডেলিভারি / সেবা) | **1,000** | 600 | 300 | 100 | Shipping speed, seller courtesy, returns |
| **Total Corpus** | **8,755** | **5,250** | **2,800** | **705** | Real-world consumer distribution |

---

## 🔍 Linguistic Error Analysis

<p align="center">
  <img src="figures/error_type_distribution.png" alt="Distribution of Error Types Across Product Categories" width="620px" />
</p>

Our fine-grained error analysis reveals 5 dominant linguistic challenges in Bangla e-commerce reviews:
1. **CDP (Context-Dependent Polarity):** Reversal of polarity depending on context (highest in Service reviews).
2. **IS (Implicit Sentiment):** Expressing satisfaction or discontent without explicit sentiment words.
3. **CNP (Complex Negation Patterns):** Multi-clause negations (e.g., *"খারাপ বলবো না, তবে..."*).
4. **SE (Sarcastic Expressions):** Backhanded compliments common in social media feedback.
5. **SV (Spelling Variations):** Informal phonetic spellings and dialectal internet slang.

---

## 📁 Repository Structure

```text
BanglaSentNet/
├── paper/
│   └── BanglaSentNet_ICDSAIA_2025.pdf           # Published Springer conference paper (pp. 283–300)
├── presentation/
│   ├── BanglaSentNet_Presentation.pdf           # Official conference presentation slides (PDF)
│   ├── BanglaSentNet_Presentation.pptx          # Editable presentation deck (PPTX)
│   └── ICDSAIA_2025_Conference_Certificate.pdf   # Official ICDSAIA 2025 presentation certificate
├── figures/
│   ├── banglasentnet_architecture.png           # End-to-end framework architecture diagram (Fig. 4)
│   ├── data_collection_pipeline.png             # Systematic data collection pipeline (Fig. 1)
│   ├── data_preprocessing_pipeline.png          # 2-stage text cleaning & normalization (Fig. 2)
│   ├── platform_distribution.png                # Review distribution across platforms (Fig. 3)
│   ├── error_type_distribution.png              # Error analysis breakdown across aspects (Fig. 5)
│   └── confusion_matrix.png                     # BanglaSentNet test set confusion matrix (Fig. 6)
├── supplementary/
│   └── Paper_Review_Response.pdf                # Peer-review author rebuttal & revision notes
├── .gitignore                                   # Standard exclusions (caches, lock files, private docs)
├── CITATION.cff                                 # GitHub native 1-click citation metadata
├── LICENSE                                      # MIT License with academic citation clause
└── README.md                                    # Professional research showcase documentation
```

---

## 📑 Citation & Academic Reference

If you find this research, dataset insights, or methodology helpful, please cite our published paper:

### BibTeX
```bibtex
@inproceedings{islam2025banglasentnet,
  author    = {Islam, Ariful and Hossen, Md Rifat},
  editor    = {Palaiahnakote, Shivakumara and Palit, Rajesh and Saraee, Mo and Atrey, Pradeep K. and Bai, Xiang and Raman, Balasubramanian},
  title     = {BanglaSentNet: A Hybrid Deep Learning Framework for Multi-Aspect Sentiment Analysis in Bangla E-Commerce Reviews},
  booktitle = {Data Science, AI and Applications (ICDSAIA 2025)},
  series    = {Communications in Computer and Information Science},
  volume    = {2682},
  pages     = {283--300},
  year      = {2025},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  isbn      = {978-3-032-11352-8},
  doi       = {10.1007/978-3-032-11352-8_20},
  url       = {https://link.springer.com/chapter/10.1007/978-3-032-11352-8_20}
}
```

### APA
> Islam, A., & Hossen, M. R. (2025). BanglaSentNet: A Hybrid Deep Learning Framework for Multi-Aspect Sentiment Analysis in Bangla E-Commerce Reviews. In *Data Science, AI and Applications* (ICDSAIA 2025), Communications in Computer and Information Science, vol. 2682, pp. 283–300. Springer, Cham. https://doi.org/10.1007/978-3-032-11352-8_20

---

## 👨‍💻 Authors & Affiliations

* **Ariful Islam** *(Corresponding Author)*  
  Department of Computer Science and Engineering  
  Chittagong University of Engineering and Technology (CUET), Chittagong, Bangladesh  
  📧 Email: [arifulislamnayem11@gmail.com](mailto:arifulislamnayem11@gmail.com)

* **Md Rifat Hossen**  
  Department of Computer Science and Engineering  
  Chittagong University of Engineering and Technology (CUET), Chittagong, Bangladesh  
  📧 Email: [rifat8851@gmail.com](mailto:rifat8851@gmail.com)

---

## ⚖️ License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Academic and commercial reuse is welcomed with citation.
