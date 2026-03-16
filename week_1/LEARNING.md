Below is a **tight, systems-first weekly plan for Weeks 1–8** (≈20 hrs/week) with **specific deliverables** and **only the study material that actually matters** for DeepMind / Anthropic trajectories.

This plan assumes:

* Strong systems background
* Minimal DL intuition
* Goal = *understand + debug + explain*, not memorize

---

# Weekly Time Budget (20 hrs)

* **10 hrs** → Build & experiment
* **6 hrs** → Read (papers / docs)
* **4 hrs** → Write (notes, blog, README)

Writing is **mandatory** — it forces understanding.

---

# WEEK 1 — Autograd Is a Lie (DL from First Principles)

### Goal

Understand **what backprop actually does** and why systems choices matter.

---

## Build (10h)

**Project: Scalar & Vector Autodiff (No PyTorch)**

* Implement:

  * Forward pass
  * Manual backward pass
* Operations:

  * Add, mul, matmul
  * ReLU
* Verify gradients numerically (finite differences)

**Deliverable**

* `autograd_scratch/`
* Test showing gradient correctness

---

## Read (6h)

* Karpathy: *Micrograd* (code, not video)
* Goodfellow DL Book — Ch 6 (Backprop)
* PyTorch Autograd docs (conceptual)

📌 Focus: *why backward is reverse graph traversal*

---

## Write (4h)

* Blog: **“What Autograd Actually Does (and Doesn’t)”**
* README explaining backward pass flow

---

# WEEK 2 — Linear Layers, Initialization, and Instability

### Goal

Understand **why models blow up or die silently**.

---

## Build (10h)

**Project: Linear Layer + Optimizer**

* Implement:

  * Linear layer forward/backward
  * SGD + Adam
* Train:

  * Linear regression
  * Small MLP on synthetic data

**Experiments**

* Bad initialization → divergence
* Learning rate sweeps

---

## Read (6h)

* Xavier / He initialization papers
* *On the Difficulty of Training Deep FFNs*
* PyTorch optimizer source (Adam)

---

## Write (4h)

* Blog: **“Why Bad Initialization Breaks Training”**
* Plot loss curves with commentary

---

# WEEK 3 — Attention from Scratch (No Magic)

### Goal

Truly understand **QKV attention mechanics**.

---

## Build (10h)

**Project: Self-Attention Layer**

* Implement:

  * Q, K, V projections
  * Scaled dot-product attention
  * Masking
* Backprop manually or semi-manual

**Tests**

* Shape sanity
* Gradient checks

---

## Read (6h)

* *Attention Is All You Need*
* Annotated Transformer blog (Jay Alammar)
* FlashAttention intro (high-level only)

📌 Ignore performance for now — correctness first.

---

## Write (4h)

* Diagram attention dataflow
* README: *Where numerical instability comes from*

---

# WEEK 4 — Build a Minimal Transformer

### Goal

Assemble parts into a **working GPT-like model**.

---

## Build (10h)

**Project: Mini-GPT**

* Components:

  * Token embedding
  * Attention
  * FFN
  * LayerNorm / RMSNorm
* Train on:

  * TinyStories or character-level text

**Required**

* Loss decreases
* No NaNs

---

## Read (6h)

* GPT-2 paper
* RMSNorm paper
* LayerNorm vs BatchNorm analysis

---

## Write (4h)

* Blog: **“Why Transformers Are Stable (Until They Aren’t)”**
* Failure modes observed

---

# WEEK 5 — Optimization & Precision (Where Systems Bite)

### Goal

Understand **mixed precision, stability, and optimizer dynamics**.

---

## Build (10h)

* Add:

  * FP16/BF16
  * Gradient scaling
* Break training intentionally
* Fix it

---

## Read (6h)

* NVIDIA AMP docs
* *Mixed Precision Training* paper
* DeepMind optimization notes (if available)

---

## Write (4h)

* Postmortem: **“How FP16 Broke My Transformer”**
* Table of fixes vs symptoms

---

# WEEK 6 — Dataloading, Throughput, and Memory

### Goal

See DL through **systems performance lens**.

---

## Build (10h)

* Custom dataloader
* Measure:

  * Tokens/sec
  * Memory usage
* Add:

  * Gradient accumulation
  * Activation checkpointing

---

## Read (6h)

* PyTorch DataLoader internals
* Activation checkpointing paper
* CUDA memory hierarchy overview

---

## Write (4h)

* Blog: **“Where Training Time Actually Goes”**
* Throughput breakdown chart

---

# WEEK 7 — Scaling Laws & Experiment Design

### Goal

Think like a **research engineer**, not a tinkerer.

---

## Build (10h)

* Train models at:

  * Multiple sizes
  * Fixed compute budget
* Plot loss vs parameters

---

## Read (6h)

* Chinchilla paper
* DeepMind scaling law papers
* “Rules of Machine Learning” (Google)

---

## Write (4h)

* Blog: **“Small Models, Big Lessons”**
* Experimental design rationale

---

# WEEK 8 — Consolidation & Public Signal

### Goal

Package work like a **lab-quality project**.

---

## Build (10h)

* Clean repo:

  * README
  * Repro scripts
  * Config files
* Add benchmark table

---

## Read (6h)

* DeepMind / Anthropic engineering blogs
* Read one recent training infra paper

---

## Write (4h)

* Capstone post: **“Training a Transformer from Scratch: A Systems View”**
* Share publicly

---

# End of Week 8 — What You Should Be Able to Do

You can:

* Explain backprop clearly
* Debug training collapse
* Reason about precision & memory
* Discuss scaling tradeoffs
* Talk like an AI systems engineer

This is **already interview-level** for L5+ AI infra roles.

---

# Next Step (After Week 8)

You’ll be ready for:

* Distributed training
* ZeRO / FSDP
* Triton kernels
* JAX (DeepMind signal)

If you want, I can next:

* Design Weeks 9–20 (distributed systems)
* Propose your **flagship Megatron-style project**
* Map this directly to **Anthropic interview loops**

Just say which one you want.

