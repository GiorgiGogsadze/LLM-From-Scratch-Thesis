# Build a Large Language Model from Scratch

**Institution:** Kutaisi International University | BSc Computer Science, 2026
**Authors:** Giorgi Gogsadze & Revaz Goguadze
**Supervisor:** Prof. Dr. Walter F. Tichy

## 📌 Project Overview
This repository contains the complete implementation of a Large Language Model (LLM) built and trained from scratch. Starting from foundational mathematical operations and raw text processing, this project constructs a GPT-2 style decoder-only transformer. Furthermore, it shifts into applied machine learning by fine-tuning the model for sentiment classification and sequential instruction following (academic writing enhancement), proving that specialized LLMs can be developed efficiently on local hardware.

## 👥 Contributions & Division of Labor
- **Joint Work:** Researching LLM designs, coding tokenization and attention mechanisms, structural implementation of the GPT architecture, and pretraining the model.
- **Giorgi Gogsadze:** Developed the sentiment classification fine-tuning pipeline.
- **Revaz Goguadze:** Designed the instruction fine-tuning framework for academic writing.

---

## 🏗️ Phase 1: Foundational Architecture (GPT-2 Style)
The base model was built entirely from scratch, utilizing PyTorch to implement the core mechanics of modern LLMs.

### 1. Data Preprocessing & Embeddings
- **Tokenization:** Integrated OpenAI's `tiktoken` Byte Pair Encoding (BPE) to efficiently break down raw text into a vocabulary of 50,257 subword tokens.
- **Data Loader:** Designed a custom PyTorch `DataLoader` employing a sliding window approach with adjustable context lengths and strides to generate input-target pairs.
- **Embeddings:** Combined learnable token embeddings with absolute positional embeddings to encode semantic meaning and word order.

### 2. Attention Mechanism
- **Multi-Head Causal Self-Attention:** Developed parallel attention heads computing scaled dot-product attention using Query, Key, and Value matrices.
- **Causal Masking:** Implemented lower-triangular masking (replacing future token scores with `-inf`) to ensure autoregressive next-token prediction.
- **Regularization:** Applied dropout mechanisms to attention weights to prevent overfitting.

### 3. Transformer Blocks & Model Assembly
- **Architecture:** Stacked 12 transformer blocks utilizing Pre-Layer Normalization.
- **Feed-Forward Networks:** Utilized GELU activation functions, which offer smoother optimization compared to standard ReLU.
- **Residual Connections:** Added shortcut connections around the attention and feed-forward sub-layers to combat vanishing gradients in deep networks.
- **Text Generation:** Implemented versatile decoding strategies, including greedy decoding, temperature scaling, and top-k sampling for dynamic text generation.

### 4. Pretraining Pipeline
- **Optimization:** Trained using the AdamW optimizer with decoupled weight decay.
- **Learning Rate Scheduling:** Applied learning rate warmup followed by cosine decay to ensure stable convergence.
- **Stability:** Utilized gradient clipping to prevent exploding gradients during deep network optimization.

---

## 🚀 Phase 2: Parameter-Efficient Fine-Tuning

### Project A: Supervised Sentiment Classification
Adapted the base language model to perform 3-class sentiment analysis (Positive, Neutral, Negative) on a balanced dataset of ~31,000 Twitter posts.
- **Architectural Shift:** Replaced the language modeling next-token output layer with a specialized classification head that outputs three logits, mapped to the final token in the sequence.
- **Data Handling:** Implemented sequence padding (to 70 tokens) to handle variable-length social media posts effectively, avoiding the need for 1024-token padding which severely degraded accuracy.
- **Partial Fine-Tuning:** Froze the lower layers of the model, training only the final transformer block, final layer normalization, and classification head.
- **Results:** Achieved a highly robust test accuracy of **74.54%** in just under 23 minutes of training, demonstrating an optimal trade-off between efficiency and effectiveness.

### Project B: Sequential Instruction Fine-Tuning (Academic Writing Assistant)
Transformed the 355M parameter GPT-2 Medium model into an academic rewriting assistant.
- **Methodology (LoRA):** Implemented Low-Rank Adaptation (LoRA) with a rank of 16 and alpha of 32. This kept foundational weights frozen while learning task-specific updates via smaller matrices, vastly reducing memory overhead.
- **Sequential Training Design:** Rather than mixing tasks, the model was trained sequentially, passing checkpoints from one stage to the next:
  - **Stage 1 (Grammar Correction):** Specialized in fixing grammatical errors. Achieved a GLEU score of **72.56** and BLEU of **60.88**.
  - **Stage 2 (Naturalness Adaptation):** Merged naturalness rewrite pairs with grammar data. Achieved the best naturalness perplexity score of **10.537**.
  - **Stage 3 (Academic Style):** Final tuning on GYAFC and ParaSCI datasets. Reached a highly optimized academic perplexity of **3.589**.
- **Analysis on Catastrophic Forgetting:** The sequential design successfully created distinct task specialists at each stage but revealed the phenomenon of catastrophic forgetting (objective interference), where later stages slightly regressed performance on earlier tasks. LoRA helped mitigate total knowledge loss.

---

## 📊 Conclusion
This thesis bridges the gap between theoretical deep learning and practical, resource-constrained deployment. By implementing the transformer architecture from scratch and leveraging advanced techniques like LoRA and partial weight freezing, the project successfully delivered highly specialized AI models capable of complex natural language tasks on local hardware.
