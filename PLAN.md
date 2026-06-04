# PLAN.md — SecureDSC reproduction + improvements

Reproduction of **"Secure Transmission in Wireless Semantic Communications With
Adversarial Training"** (SecureDSC, Shi et al., IEEE Comm. Letters 2025), in
PyTorch, plus two improvements:

- **Improvement A (headline):** CSI-based neural key generation (replace the
  random session key with a key derived from the reciprocal physical channel).
- **Improvement B (supporting):** adaptive `lambda` scheduler driven by the
  training loss gap.

Everything is config-driven and toggleable so we can run any combination in an
A/B ablation grid.

---

## 1. System overview (what we reproduce)

A DeepSC-style **text** semantic communication system with three parties:

- **Alice** — transmitter.
- **Bob** — legitimate receiver, shares the session key with Alice, jointly
  trained with Alice.
- **Eve** — eavesdropper. Same network *shape* as Bob but **not** jointly
  trained with Alice; uses a *wrong* key (random key in the baseline, or her own
  CSI-derived key in Improvement A). She taps the same channel output and trains
  her own decoder to do her best (adversary).

### Forward pipeline (Bob path)
```
tokens
  -> Embedding (+positional)
  -> Semantic Encoder        (Transformer encoder stack)
  -> Encryptor               (Transformer decoder: features attend to KEY tokens)
  -> Channel Encoder         (dense, -> channel symbols, power-normalized)
  -> Wireless Channel        (y = h*x + n ; AWGN or Rayleigh)
  -> Channel Decoder         (dense, -> d_model features)
  -> Decryptor               (Transformer decoder: ciphertext attends to KEY tokens)
  -> Semantic Decoder        (Transformer encoder stack)
  -> Prediction layer        (Linear -> softmax over vocab)
  -> m_hat
```
A **Key-Processing Network** maps the raw key vector to `n_key_tokens × d_model`
"key tokens" that the encryptor/decryptor cross-attend to. With the correct key
the decryptor inverts the encryptor; with a wrong key it cannot.

**Eve's path** reuses the same channel output `y` but with her *own*
channel-decoder, decryptor, semantic-decoder and prediction layer, fed by her
*own* (wrong) key.

### Losses
- `L_sem` — semantic autoencoder CE: `tokens -> embed -> semantic enc -> semantic
  dec -> prediction`, bypassing cipher+channel. Keeps the semantic codec faithful.
- `L_cip` — cipher reconstruction MSE: `features -> encryptor -> (channel) ->
  decryptor` should return the features. Keeps the cipher codec invertible *given
  the key*.
- `L_B = CE(m, m_hat)` — Bob, full path.
- `L_E = CE(m, m_bar)` — Eve, full path with her wrong key.
- `L_joint = L_B + |L_E - lambda|`. Minimised over Alice/Bob params: drives Bob's
  loss down while pulling Eve's *achievable* loss toward `lambda` (a high target
  = Eve stays confused).

### Training — Algorithm 1 (per batch, cyclic)
1. Update **semantic enc/dec** with `L_sem`.
2. Update **encryptor/decryptor (+key-proc)** with `L_cip`.
3. Update the **whole Alice/Bob network** with `L_joint` (Eve params in-graph for
   the `|L_E-lambda|` term but **not** stepped here — gradient still flows back to
   Alice so she learns to confuse Eve).
4. Update **Eve's network** with `L_E` (Eve does her best — adversary step).

### Evaluation
- BLEU 1–4-gram for Bob and Eve, swept over SNR (AWGN and Rayleigh).
- Sanity target (paper): Bob ≈ 0.9+ BLEU, Eve ≲ 0.2.

---

## 2. Improvements

### A. CSI-based neural key generation
- **ReciprocalChannel** emits per-party CSI estimates such that `h_AB ≈ h_BA`
  (reciprocal, correlation coefficient `rho`, independent estimation noise), while
  Eve's `h_AE`, `h_BE` are **independent** of the Alice–Bob channel.
- **CSIKeyGenerator** — small MLP mapping a party's CSI estimate -> a key vector.
  Trained so Alice's and Bob's keys **agree** (low mismatch) while Eve's derived
  key does **not**. Loss: `MSE(k_A, k_B)` pulled down + a margin term pushing
  `k_E` away (or just measured, since Eve is independent by construction).
- **Reconciliation / quantization** — sign-quantize key vectors to bits; an
  information-reconciliation step (parity/majority over repeated bits) makes Alice
  and Bob converge to an *identical* bitstring. Reported: Alice–Bob bit-agreement
  rate, Eve mismatch.
- The agreed key replaces the random key into the encryptor/decryptor.

### B. Adaptive lambda scheduler
- `FixedLambda` (baseline) returns a constant.
- `AdaptiveLambda` tracks an EMA of `L_E` (and the `L_E - L_B` gap) and moves
  `lambda` each step via a feedback rule toward a *target Eve loss*
  (`target_frac * ln(vocab)`), clipped to `[0, ln(vocab)]`:
  `lambda <- clip(lambda + eta * (target_E - ema_L_E))`.
  This removes manual tuning of the absolute `lambda` (you specify a target
  difficulty instead) and stabilises training.

---

## 3. Modular interfaces (base class + impls, config-selected)

- `KeyGenerator` (base): `RandomKeyGenerator` | `CSIKeyGenerator`.
- `LambdaScheduler` (base): `FixedLambda` | `AdaptiveLambda`.
- `Channel` (base): `AWGNChannel` | `RayleighChannel`; plus `ReciprocalChannel`
  helper emitting correlated CSI (Alice/Bob) and independent CSI (Eve),
  parameterised by `rho` and SNR.
- `SecureDSC` model, the Algorithm-1 trainer, and the evaluator consume these via
  config so any combination runs.

---

## 4. Dataset

- **Default:** a subset of an English corpus (Europarl `en`). Loader takes a path
  or downloads; sentences filtered by length, whitespace-tokenised, vocab built
  with `<pad> <start> <end> <unk>`, padded to `max_len`.
- **Smoke test:** a small built-in toy corpus (a few dozen English sentences) with
  a tiny vocab so the whole pipeline runs on CPU in seconds.

---

## 5. Defaults / decisions (chosen for ambiguous points)

| Choice | Default | Rationale |
|---|---|---|
| `d_model` | 128 | DeepSC uses 128. |
| `nhead` | 8 | standard. |
| layers (each block) | 3 sem / 2 cipher | DeepSC ~3; cipher lighter. |
| `channel_dim` per token | 16 | DeepSC-style compression. |
| `max_len` | 30 tokens | typical DeepSC sentence length. |
| key length `L_key` | 64 | enough bits for the cipher tokens. |
| `n_key_tokens` | 4 | key-proc reshapes 64 -> 4×d_model via MLP. |
| cipher/decoder type | Transformer **decoder** w/ cross-attn to key tokens | gives a real correct-key vs wrong-key functional gap. |
| decoders | **non-autoregressive** (predict whole sentence at once) | matches DeepSC; avoids AR decoding cost. |
| channel symbols | real vectors, power-normalised to unit avg power | SNR added in real domain (complex split optional). |
| `lambda` baseline | 0.5 | mid-range; paper sweeps it. |
| adaptive target | `0.9 * ln(vocab)` Eve loss | keep Eve near-random. |
| optimizer | Adam, lr 1e-4 (joint), 1e-3 (sub-steps) | DeepSC-ish. |
| BLEU | custom 1–4-gram w/ smoothing, weights (.25,.25,.25,.25) | self-contained + unit-tested. |

These are recorded here rather than blocking; tune in YAML.

---

## 6. Repository layout

```
securedsc/
  config.py        dataclass configs + YAML load/merge
  utils.py         seeding, device, logging
  data.py          vocab, toy corpus, Europarl loader, DataLoader
  channels.py      Channel base, AWGN, Rayleigh, ReciprocalChannel
  keygen.py        KeyGenerator base, Random, CSI, reconciliation
  lambda_sched.py  LambdaScheduler base, Fixed, Adaptive
  modules.py       embedding/positional, transformer blocks, channel codec, prediction
  model.py         SecureDSC (Alice/Bob + Eve), forward paths, loss heads
  metrics.py       BLEU 1-4, key agreement
  train.py         Algorithm-1 trainer
  eval.py          BLEU-vs-SNR, correlation sweep, ablation
scripts/
  run_train.py     train one config
  run_eval.py      evaluate a checkpoint -> results/
  run_ablation.py  {random,csi} x {fixed,adaptive} grid + plots
configs/
  smoke.yaml       tiny CPU smoke test
  base_awgn.yaml   full baseline (random key, fixed lambda, AWGN)
tests/             channel math, reciprocal CSI corr, key agreement, lambda rule, BLEU
results/           metrics tables + plots (runtime)
```

## 7. Delivery order
- **P0** scaffold + PLAN (this).
- **P1** baseline (random key, fixed lambda, AWGN), Algorithm-1, BLEU-vs-SNR;
  confirm Bob-high / Eve-low.
- **P2** AdaptiveLambda + A/B vs Fixed (curves, final gap, multi-seed stability).
- **P3** ReciprocalChannel + CSIKeyGenerator + reconciliation; key agreement /
  Eve mismatch.
- **P4** full eval: ablation grid, Rayleigh, correlation sweep, plots + tables.
