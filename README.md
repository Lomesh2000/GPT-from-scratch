# GPT-from-scratch
# 🧠 Mini-GPT – Character-Level Transformer on Shakespeare Data

A lightweight GPT-style decoder-only transformer trained from scratch on Shakespeare text.  
Implements: tokenization, transformer blocks, self-attention, positional embeddings, training loop, and text generation — all in pure PyTorch.

---

## 🚀 Features
- ✓ Custom character-level tokenizer (saved + reloadable)
- ✓ Decoder-only Transformer architecture (GPT-mini)
- ✓ Trained on Shakespeare dataset
- ✓ Generate text from a starting prompt
- ✓ Model + tokenizer saving & loading
- ✓ TensorBoard metrics logging

---

## 🏗️ Model Architecture
| Component | Details |
|----------|---------|
| Type | Decoder-only Transformer |
| Layers | N attention blocks (configurable) |
| Embedding Dim | 384 |
| Attention Heads | multi-head |
| Optimizer | AdamW |
| Loss | Cross-Entropy |

## Hyperparamters
```python
batch_size    = 64
block_size    = 256
max_iters     = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters    = 200
n_embd        = 384
n_head        = 6
n_layer       = 6
dropout       = 0.2
device        = 'cuda' if torch.cuda.is_available() else 'cpu'
random_seed   = 1337
```

---

## 📊 Training Metrics

### 🔁 Training Loss Curve
![Training Loss](losses_curves/train_loss.png)

### 📉 Validation Loss Curve
![Validation Loss](losses_curves/val_loss.png)

### 🪜 Per-Step Training Loss (Batch-level change)
![Per Step Loss](losses_curves/train_step_loss.png)

> Loss steadily decreases → model is learning patterns from text.

---

## 🧪 Text Generation Example

