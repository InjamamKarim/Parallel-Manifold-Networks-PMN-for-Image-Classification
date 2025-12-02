# Parallel-manifold-net
A parameter-efficient, deterministic deep learning alternative based on 2D manifold representations. Achieves ~52% Top-1 / ~81% Top-5 on CIFAR-100 with &lt;1M parameters and ~10 minutes of training.



# Parallel Manifold Networks (PMN)

This repo contains a minimal reference implementation of **Parallel Manifold Networks**
– a small, deterministic image classifier that reaches **~52% Top-1 / ~81% Top-5 on CIFAR-100**
with **< 1M parameters** and **~10 minutes of training on a single GPU**.

Core ideas:

- Representations live on many small **2D manifolds ("universes")** instead of high-D feature vectors.
- Each manifold uses **radius** (‖z‖) and **phase** (angle) as two complementary signals.
- Two parallel paths (radius / cos-phase) are stacked, then fused for classification.
- The final linear classifier can be strongly **pruned / regularized per universe**.
- The architecture is **small, modular, embedded-friendly and close to deterministic**.

This code trains PMN on CIFAR-100 from scratch.

---

## Quickstart

```bash
git clone https://github.com/<your-user>/parallel-manifold-net.git
cd parallel-manifold-net

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
