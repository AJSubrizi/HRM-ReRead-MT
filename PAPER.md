# HRM-ReRead-MT: Hierarchical Re-Reading with Multi-Teacher Latent Consolidation

**Antonio J. Subrizi**  
Founder & Lead Engineer, NailDash  
Independent AI Researcher  
Italy  
[ajsubrizi.dev](https://ajsubrizi.dev) | antoniosubrizi17@outlook.it

## Abstract

HRM-ReRead-MT is a proposed training paradigm that combines hierarchical recurrent reasoning,
iterative re-reading, latent-state consolidation, and teacher-guided distillation. The core hypothesis is
that a smaller student model can improve reasoning quality by revisiting high-quality examples across
multiple passes while carrying forward a high-level latent memory.

This repository currently provides a mathematical formulation and a small executable reference scaffold.
It does not yet provide full-scale empirical validation or a production HRM training stack.

## 1. Introduction

Large language models are powerful but expensive to train and adapt. Recurrent and hierarchical
reasoning models suggest an alternate direction: instead of relying only on larger datasets and wider
models, a student can refine internal state over repeated passes.

HRM-ReRead-MT adds two ideas to that direction:

1. **Iterative Re-Reading** with latent consolidation across passes.
2. **Teacher-guided Distillation** from a stronger model into the student training signal.

The result is a research hypothesis: repeated, teacher-guided passes may improve data efficiency and
reasoning depth when compared with ordinary single-pass fine-tuning.

## 2. Related Work

- Hierarchical and recurrent reasoning models.
- Re-reading, refinement, and self-correction loops.
- Knowledge distillation and teacher-student training.
- Latent-space memory and consolidation.

## 3. Proposed Method

### 3.1 HRM Backbone

The intended backbone uses two coupled recurrent modules:

- **H-module**: high-level, slower latent planning.
- **L-module**: low-level, faster token/local computation.

For each forward pass:

\[
\begin{cases}
z_L^{(t)} = L(z_L^{(t-1)}, z_H^{(k-1)}) & t = 1 \dots T \\
z_H^{(k)} = H(z_L^{(T)}, z_H^{(k-1)}) & k = 1 \dots K
\end{cases}
\]

### 3.2 Re-Reading with Latent Consolidation

During each re-read epoch \( e = 1 \dots N \):

- The model performs a forward pass.
- For \( e > 1 \), the previous high-level state \( z_H^{e-1} \) is loaded from a memory buffer.
- A consolidation term encourages useful continuity between passes.

\[
\mathcal{L}_\text{cons} = \|z_H^e - z_H^{e-1}\|_2^2
\]

### 3.3 Teacher-Guided Distillation

Where teacher logits or hidden states are available, the desired distillation objective is:

\[
\mathcal{L}_\text{distill} =
\text{KL}(\text{softmax}(f_\theta / \tau) \parallel \text{softmax}(f_\text{teacher} / \tau))
+ \lambda \| \text{proj}(z_H) - z_H^\text{teacher} \|_2^2
\]

Many hosted teacher APIs expose generated text but not hidden states or full logits. In those cases,
the current scaffold stores teacher responses for supervised or preference-style follow-up training,
while leaving latent/logit distillation as future provider-dependent work.

Total intended loss:

\[
\mathcal{L} = \mathcal{L}_\text{pred} + \alpha \mathcal{L}_\text{cons} + \beta \mathcal{L}_\text{distill}
\]

### 3.4 Reference Scaffold

The repository includes:

- `src/hrm_reread_mt/data_augment.py`: JSONL teacher augmentation.
- `src/hrm_reread_mt/train_re_read_mt.py`: a small PyTorch loop demonstrating latent memory reuse.
- `tests/test_reference_training.py`: a local smoke test.

The included model is intentionally tiny and character-level. Its purpose is to make the repository
testable, not to claim benchmark performance.

## 4. Expected Improvements

The following are hypotheses to test, not measured results:

| Benchmark | Baseline | HRM-ReRead-MT projected | Status |
| --- | --- | --- | --- |
| MATH Pass@1 | TBD | improvement expected | unvalidated |
| MMLU 5-shot | TBD | improvement expected | unvalidated |
| ARC-Challenge | TBD | improvement expected | unvalidated |
| HumanEval | TBD | improvement expected | unvalidated |

Any future benchmark table should include the exact base model, dataset, compute budget, evaluation
harness, random seeds, and confidence intervals.

## 5. Limitations

- This is a theoretical proposal with an executable reference scaffold.
- No empirical performance claims are currently established.
- The current training code is a smoke-test implementation, not a full HRM reproduction.
- Teacher hidden-state distillation depends on provider support and is not implemented in the public API client.

## 6. Roadmap

- Select and integrate a concrete HRM backbone.
- Add tokenizer-backed datasets and batching.
- Add checkpointing, configs, evaluation scripts, and experiment tracking.
- Add teacher-logit or hidden-state distillation where technically available.
- Publish reproducible benchmark results.

## Citation

```bibtex
@misc{subrizi2026hrmrereadmt,
  author = {Antonio J. Subrizi},
  title = {HRM-ReRead-MT: Hierarchical Re-Reading with Multi-Teacher Latent Consolidation},
  year = {2026},
  note = {Theoretical proposal and reference scaffold},
  url = {https://github.com/AJSubrizi/HRM-ReRead-MT}
}
```

## License

MIT.
