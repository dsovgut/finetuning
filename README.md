# Fine‑Tuning **Mistral‑7B** with **QLoRA**

> *Gen AI (CMSC 37815) — UChicago MS‑ADS*

---

## 📚 Overview

This repo contains the notebook and lightweight LoRA adapter I produced for the course assignment:

1. **Objective** – use **QLoRA** (4‑bit quantisation + LoRA adapters) to instruction‑tune an open‑weights LLM.
2. **Setup** – base model **`mistralai/Mistral‑7B‑v0.1`**, trained for **1 epoch** on **1 000** examples from the *Alpaca* instruction dataset.
3. **Deliverables** – notebook (`Finetuning_Danylo.ipynb`).

All hyper‑parameters, code, and evaluation logic follow the template notebook shared in class with only minor modifications (see below).

---

## 🔧 Key Choices

| Component        | Value                                                   | Why                                                          |
| ---------------- | ------------------------------------------------------- | ------------------------------------------------------------ |
| **Base model**   | `mistralai/Mistral‑7B‑v0.1`                             | Strong reasoning, HF‑native, LoRA‑friendly                   |
| **Dataset**      | `tatsu‑lab/alpaca` – first 1 000 rows                   | Clean instruction–response pairs; fits course compute budget |
| **Quantisation** | 4‑bit GPTQ                                              | Cuts memory footprint dramatically; required for QLoRA       |
| **LoRA config**  | `r=8`, `α=16`, dropout 0.05, target `q_proj` & `v_proj` | Standard for 7‑B models                                      |
| **Trainer**      | `trl.SFTTrainer`                                        | Matches lecture notebook for reproducibility                 |

---

## 📈 Evaluation

The notebook computes three automatic metrics on a held‑out 100‑example slice:

| Metric          | Score  |
| --------------- | ------ |
| **BLEU**        | 0.5093 |
| **ROUGE‑L F1**  | 0.6298 |
| **Exact Match** | 0.0400 |

*(See the last notebook cell for the raw computation.)*

### Example Generations

<details><summary>Click to expand</summary>

```text
Instruction: List three creative uses for a paper clip.

Model:  
1. Mini phone stand for video calls.  
2. Emergency zipper pull.  
3. Ear‑bud cable tidy.
```

```text
Instruction: Explain a hash table to a 12‑year‑old.

Model:  
Imagine millions of numbered mailboxes. A magic math trick tells you exactly which box to drop your note in, so you can find it instantly later.
```

</details>

---

## Notes

* QLoRA lets us fine‑tune large models on commodity GPUs by freezing base weights and training only rank‑reduced adapters on a 4‑bit quantised backbone.
* Automatic metrics only approximate quality; manual spot‑checks are essential.
* Feel free to open an issue if you hit a snag reproducing the run.

---

## Acknowledgements
* Course materials by **Prof. Nousetouane** (Gen AI, Spring 2025).
