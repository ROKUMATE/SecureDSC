# SecureDSC — Secure Wireless Semantic Communication with Adversarial Training

PyTorch reproduction of **"Secure Transmission in Wireless Semantic
Communications With Adversarial Training"** (SecureDSC, Shi et al., IEEE Comm.
Letters 2025), plus two improvements:

- **Improvement A (headline) — CSI-based neural key generation.** The random
  session key is replaced by a key *derived from the reciprocal physical channel*
  (CSI). Alice and Bob observe correlated CSI and converge to an identical key
  through quantisation + reconciliation; Eve's CSI is independent so her key does
  not match.
- **Improvement B (supporting) — adaptive `lambda` scheduler.** The fixed
  secrecy hyperparameter `lambda` is replaced by a feedback controller driven by
  the training loss gap, removing manual tuning and stabilising training.

Every component is behind a base class and selected via YAML config, so any
combination runs (the ablation grid is `{random, csi} × {fixed, adaptive}`).

See [PLAN.md](PLAN.md) for the architecture, the loss/training details
(Algorithm 1), and the default choices.

---

## 1. Install (on any machine)

Requires **Python 3.11–3.13** (verified on 3.13; torch wheels for 3.14 may lag).

```bash
cd p-2
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

On a CUDA machine, `pip install torch==2.12.0` already pulls the GPU build; set
`train.device: auto` (the default) and it will use the GPU automatically. Apple
Silicon falls back to MPS, otherwise CPU.

Verify the install with the test suite (≈1 s, CPU):

```bash
pytest -q
```

---

## 2. Quick smoke test (CPU, seconds)

Proves the whole pipeline wires together on a tiny model + toy corpus:

```bash
python -m scripts.run_train  --config configs/smoke.yaml
python -m scripts.run_eval   --ckpt   results/smoke/ckpt.pt --num-batches 2
python -m scripts.run_ablation --config configs/smoke.yaml --seeds 0 1 --epochs 1
```

(The smoke config is intentionally too small to reach good BLEU — it only checks
that training, evaluation, the CSI key sweep, plots and tables all run.)

---

## 3. Get the dataset first

> **Important:** real Bob-high / Eve-low secrecy needs a high-entropy corpus.
> The built-in **toy corpus is only 20 sentences** — Eve simply *memorises* all
> messages, so you will see `BLEU(Eve) ≈ BLEU(Bob)`. Use **Europarl** for any
> result you report.

```bash
python -m scripts.get_europarl                 # -> data/europarl-v7.en (~200 MB download)
# or cap it:  python -m scripts.get_europarl --max-lines 200000
```

This downloads the Europarl v7 fr-en release and extracts its English side (the
corpus DeepSC uses). The loader lower-cases and strips punctuation automatically.

## 4. Reproduce the baseline (P1) and the improvements

All real runs use `configs/base_europarl.yaml` (DeepSC-sized model, real corpus,
corpus-appropriate `lambda`). GPU strongly recommended.

```bash
# P1 — baseline: random key, fixed lambda, AWGN
python -m scripts.run_train --config configs/base_europarl.yaml --run-name base_europarl
python -m scripts.run_eval  --ckpt results/base_europarl/ckpt.pt
# -> results/base_europarl/bleu_vs_snr.png   (expect BLEU(Bob) high, BLEU(Eve) low)

# P2 — adaptive lambda A/B (compare training stability / final gap)
python -m scripts.run_train --config configs/base_europarl.yaml --lam fixed    --run-name lam_fixed
python -m scripts.run_train --config configs/base_europarl.yaml --lam adaptive --run-name lam_adaptive

# P3 — CSI-based key generation
python -m scripts.run_train --config configs/base_europarl.yaml --keygen csi --run-name csi_key
python -m scripts.run_eval  --ckpt results/csi_key/ckpt.pt
# -> results/csi_key/key_vs_correlation.png  (Alice/Bob agreement rises with rho,
#    Eve mismatch stays high)

# Rayleigh channel
python -m scripts.run_train --config configs/base_europarl.yaml --channel rayleigh --run-name base_rayleigh
```

### Full ablation grid (P4) — the headline experiment

```bash
python -m scripts.run_ablation --config configs/base_europarl.yaml --seeds 0 1 2
```

Writes to `results/ablation/`:
- `ablation_table.csv` / `.json` — mean±std over seeds of BLEU(Bob), BLEU(Eve),
  Bob−Eve gap, Alice/Bob key agreement, Eve mismatch for all four cells.
- `training_curves.png` — `L_B`, `L_E`, `lambda` per epoch (fixed vs adaptive).
- `ablation_bleu_vs_snr.png` — BLEU-vs-SNR for each cell.

---

## 5. What to look for (sanity checks from the paper)

| Signal | Expectation |
|---|---|
| BLEU(Bob) vs SNR | rises with SNR toward ~0.9+ (paper target) |
| BLEU(Eve) vs SNR | stays low (≲ 0.2) — she cannot decrypt |
| Alice/Bob key agreement (CSI) | high and increasing with CSI correlation `rho` |
| Eve key mismatch (CSI) | ~0.3–0.5, roughly flat in `rho` (independent CSI) |
| Adaptive vs fixed `lambda` | lower seed-to-seed variance, comparable/larger Bob−Eve gap |

**On `lambda` (secrecy knob):** `L_joint = L_B + |L_E − lambda|` pulls Eve's loss
*toward* `lambda`, so set `lambda` **high** (~`0.9·ln(vocab)`) to keep Eve near
random-guessing. A *low* `lambda` trains Alice to *help* Eve — there is a runtime
warning if you set it too low. Because the right value scales with the vocab, the
**adaptive** scheduler (`--lam adaptive`) is the robust choice.

---

## 6. Config reference (YAML sections)

| Section | Key | Meaning |
|---|---|---|
| `data` | `source` | `toy` \| `europarl` |
| | `path`, `max_len`, `max_vocab`, `max_sentences`, `val_fraction` | corpus / vocab |
| `model` | `d_model`, `nhead`, `dim_feedforward` | Transformer size |
| | `num_sem_layers`, `num_cipher_layers` | depth of semantic / cipher blocks |
| | `channel_dim`, `key_len`, `n_key_tokens` | channel symbols / key width / key tokens |
| `channel` | `kind` | `awgn` \| `rayleigh` |
| | `train_snr_db` | SNR used during training |
| | `csi_rho`, `csi_dim`, `csi_snr_db` | reciprocal-CSI correlation / size / estimation SNR |
| `keygen` | `kind` | `random` \| `csi` |
| | `csi_hidden`, `recon_repeat` | CSI key MLP size / repetition-code length |
| `lam` | `kind` | `fixed` \| `adaptive` |
| | `value` | fixed lambda (or adaptive initial) |
| | `target_frac`, `eta`, `ema` | adaptive: target = `target_frac·ln(vocab)`, step, EMA |
| `train` | `epochs`, `batch_size`, `lr_joint`, `lr_sub`, `lr_eve` | optimisation |
| | `device` | `auto` \| `cpu` \| `cuda` |
| | `seed`, `out_dir`, `run_name` | reproducibility / outputs |
| `eval` | `snr_db_list`, `rho_list`, `num_batches`, `bleu_ngrams` | sweeps |

CLI overrides on `run_train`: `--epochs --seed --run-name --keygen --lam --channel`.

---

## 7. Project layout

```
securedsc/
  config.py        dataclass configs + YAML load
  utils.py         seeding, device, logging, SNR<->noise
  data.py          vocab, toy corpus, corpus loader, dataloaders
  channels.py      Channel base, AWGN, Rayleigh, ReciprocalChannel (CSI)
  keygen.py        KeyGenerator base, Random, CSI, reconciliation
  lambda_sched.py  LambdaScheduler base, Fixed, Adaptive
  modules.py       embeddings, transformer/cross-attn blocks, channel codec, key-proc
  model.py         SecureDSC (Alice/Bob + Eve), 3 forward paths, param groups
  metrics.py       self-contained BLEU 1-4, key agreement
  train.py         Algorithm-1 trainer
  eval.py          BLEU-vs-SNR, correlation sweep, plots
scripts/           run_train.py, run_eval.py, run_ablation.py
configs/           smoke.yaml, base_awgn.yaml
tests/             channel math, reciprocal CSI, key agreement, lambda, BLEU
results/           checkpoints, metrics, plots (created at runtime)
```

---

## 8. Notes / decisions

- Decoders are **non-autoregressive** (predict the whole sentence at once), as in
  DeepSC. The encryptor/decryptor are Transformer **decoder** stacks that
  cross-attend to *key tokens*, so the correct key inverts the cipher and a wrong
  key fails.
- The CSI cipher path uses a *reconciled, identical* key; the reported
  `ab_agreement` is the **raw** Alice/Bob bit agreement *before* the public
  reconciliation exchange — that is the meaningful security/quality signal.
- Algorithm 1 runs four sub-steps per batch (semantic, cipher, joint, Eve) plus a
  fifth (CSI key generator) when CSI keys are enabled. See [PLAN.md](PLAN.md) §1.
- All runs are seeded; pass `--seed` (or the ablation `--seeds`) for multi-seed
  stability studies.
