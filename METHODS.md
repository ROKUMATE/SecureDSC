# METHODS — SecureDSC Reproduction and Improvements

A self-contained technical reference for writing up this project as a research
paper. It documents the system model, every module's mathematics, the training
algorithm, the two proposed improvements, the evaluation protocol, and the exact
mapping from equations to source files. Equations use the notation in §1.

> Scope: this reproduces **"Secure Transmission in Wireless Semantic
> Communications With Adversarial Training"** (SecureDSC; Shi et al., *IEEE
> Communications Letters*, 2025) and adds two contributions: (A) CSI-based neural
> key generation, and (B) an adaptive secrecy-weight (`λ`) scheduler.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $\mathbf{m}=(m_1,\dots,m_L)$ | source sentence, $L$ tokens from vocabulary $\mathcal{V}$, $|\mathcal{V}|=V$ |
| $\hat{\mathbf{m}}, \bar{\mathbf{m}}$ | Bob's and Eve's reconstructed sentences |
| $d$ | model (embedding) dimension `d_model` |
| $C$ | channel symbols per token `channel_dim` |
| $\mathbf{S}\in\mathbb{R}^{L\times d}$ | semantic features |
| $\mathbf{Z}\in\mathbb{R}^{L\times d}$ | ciphertext features (encryptor output) |
| $\mathbf{x}\in\mathbb{R}^{L\times C}$ | transmitted channel symbols (power-normalised) |
| $\mathbf{y}\in\mathbb{R}^{L\times C}$ | received channel symbols |
| $\mathbf{k}\in\{-1,+1\}^{B}$ | key bit-vector, width $B$ = `key_len` |
| $\mathbf{K}\in\mathbb{R}^{T\times d}$ | key tokens, $T$ = `n_key_tokens` |
| $h$ | channel gain; $n$ AWGN noise |
| $\rho$ | Alice–Bob CSI correlation coefficient |
| $\gamma$ | SNR (dB); $\gamma_{\text{lin}}=10^{\gamma/10}$ |
| $\lambda$ | secrecy weight in the joint loss |
| $\theta_{\text{AB}},\theta_{\text{E}}$ | Alice/Bob (legitimate) and Eve parameters |

Subscripts $A,B,E$ denote Alice, Bob, Eve. Parties share the *architecture*; Bob
is jointly trained with Alice, Eve is an independently-trained adversary using a
wrong key.

---

## 2. System model

A three-party text semantic-communication system (DeepSC-style), `securedsc/model.py`:

```
m → Embedding(+pos) → SemanticEncoder → Encryptor(·,K_A) → ChannelEncoder
    → power-normalise → Channel(γ) → y
y → ChannelDecoder → Decryptor(·,K_B) → SemanticDecoder → Prediction → m̂   (Bob)
y → ChannelDecoder_E → Decryptor_E(·,K_E) → SemanticDecoder_E → Pred_E → m̄  (Eve)
```

Bob and Eve both observe the *same* received signal $\mathbf{y}$ (Eve taps the
channel); they differ only in (i) having separate decoder-side parameters and
(ii) the key they hold.

### 2.1 Transmitter (Alice)

1. **Embedding + positional encoding** (`modules.PositionalEncoding`):
   $\mathbf{E}=\text{Embed}(\mathbf{m})+\mathbf{P}\in\mathbb{R}^{L\times d}$,
   sinusoidal $\mathbf{P}$.
2. **Semantic encoder** $f_{\text{se}}$ — a Transformer encoder stack:
   $\mathbf{S}=f_{\text{se}}(\mathbf{E})$.
3. **Key processing** $g_{\text{kp}}$ (`modules.KeyProcessingNetwork`): an MLP maps
   the key bit-vector to $T$ key tokens,
   $\mathbf{K}=g_{\text{kp}}(\mathbf{k})\in\mathbb{R}^{T\times d}$.
4. **Encryptor** $f_{\text{enc}}$ — a Transformer **decoder** stack whose
   cross-attention attends to the key tokens (`modules.CrossAttnDecoderStack`):
   $\mathbf{Z}=f_{\text{enc}}(\mathbf{S},\,\mathbf{K}_A)$.
   Cross-attention to $\mathbf{K}$ is what binds the cipher to the key: the
   transform applied to $\mathbf{S}$ is key-conditioned, so only the matching key
   inverts it.
5. **Channel encoder** + **power normalisation**: $\mathbf{x}'=h_{\text{ce}}(\mathbf{Z})$,
   then per-sample normalisation so the mean symbol power is 1:
$$
\mathbf{x}=\frac{\mathbf{x}'}{\sqrt{\tfrac{1}{LC}\sum_{l,c}\mathbf{x}'^2_{l,c}+\epsilon}}\,.
$$

### 2.2 Channel (`securedsc/channels.py`)

General model $\mathbf{y}=h\odot\mathbf{x}+\mathbf{n}$. With unit signal power the
per-real-dimension noise std for SNR $\gamma$ (dB) is
$$
\sigma_n=\sqrt{P/\gamma_{\text{lin}}}=\sqrt{10^{-\gamma/10}}\,,\qquad P=1 .
$$

- **AWGN** (`AWGNChannel`): $h=1$, $\mathbf{y}=\mathbf{x}+\mathbf{n}$,
  $\mathbf{n}\sim\mathcal{N}(0,\sigma_n^2)$.
- **Rayleigh** (`RayleighChannel`): one complex gain per sample,
  $h=(h_r+jh_i)/\sqrt2$ with $h_r,h_i\sim\mathcal N(0,1)$ so $\mathbb E|h|^2=1$.
  Channel symbols are treated as interleaved I/Q pairs and the complex gain is
  applied to each pair, then AWGN is added. The decoder learns to equalise (no
  explicit CSI given to the data path), as in DeepSC.

The channel is **differentiable**, so gradients flow end-to-end through it.

### 2.3 Receivers

- **Bob**: $\hat{\mathbf F}=h_{\text{cd}}(\mathbf y)$ (channel decoder),
  $\hat{\mathbf Z}=f_{\text{dec}}(\hat{\mathbf F},\mathbf K_B)$ (decryptor, correct
  key), $\hat{\mathbf S}=f_{\text{sd}}(\hat{\mathbf Z})$ (semantic decoder),
  logits $=W_{\text{pred}}\hat{\mathbf S}$, $\hat{\mathbf m}=\arg\max$.
- **Eve**: identical structure with **independent parameters**
  $h^E_{\text{cd}}, f^E_{\text{dec}}, f^E_{\text{sd}}, W^E_{\text{pred}}, g^E_{\text{kp}}$
  and a **wrong key** $\mathbf k_E$.

All decoders are **non-autoregressive** — the whole sentence is predicted in one
pass (DeepSC behaviour); no causal mask is used.

---

## 3. Losses

Let $\text{CE}(\mathbf m,\text{logits})=-\frac1{L'}\sum_{l:\,m_l\neq\text{pad}}\log p_\theta(m_l\mid\mathbf y)$
be token cross-entropy with padding ignored ($L'$ = non-pad tokens).

| Loss | Definition | Trains | Code |
|---|---|---|---|
| $L_{\text{sem}}$ | $\text{CE}(\mathbf m,\,\text{Pred}(f_{\text{sd}}(f_{\text{se}}(\mathbf E))))$ — semantic autoencoder, **no** cipher/channel | semantic codec | `forward_semantic` |
| $L_{\text{cip}}$ | $\lVert f_{\text{dec}}(h_{\text{cd}}(\text{Channel}(h_{\text{ce}}(f_{\text{enc}}(\bar{\mathbf S},\mathbf K_A)))),\mathbf K_B)-\bar{\mathbf S}\rVert_2^2$, with $\bar{\mathbf S}=\text{sg}[\mathbf S]$ | cipher codec | `forward_cipher` |
| $L_B$ | $\text{CE}(\mathbf m,\hat{\mathbf m}\text{ logits})$ | Bob path | `forward_full` |
| $L_E$ | $\text{CE}(\mathbf m,\bar{\mathbf m}\text{ logits})$ | Eve path | `forward_full` |
| $L_{\text{joint}}$ | $L_B+\lvert L_E-\lambda\rvert$ | whole Alice/Bob net | trainer |

$\text{sg}[\cdot]$ = stop-gradient. $L_{\text{cip}}$ uses MSE reconstruction in
feature space so the cipher codec is **invertible given the key**, independent of
the semantic codec.

**Secrecy term.** $\lvert L_E-\lambda\rvert$ pulls Eve's achievable loss *toward*
$\lambda$. For secrecy $\lambda$ must be **large** (near the random-guess loss
$\ln V$): minimising the term then keeps $L_E$ high (Eve confused) without the
instability of an unbounded $-L_E$ maximisation. A *small* $\lambda$ would train
Alice to *help* Eve — see §6.

---

## 4. Algorithm 1 — adversarial training schedule

Per mini-batch, four (optionally five) sub-steps update disjoint objective groups
with separate Adam optimisers (`securedsc/train.py`):

```
Input: batch m, channel C, train SNR γ, scheduler Λ
1.  draw key bundle (k_A, k_B, k_E)               # §5
2.  (i)   L_sem  ← CE(semantic AE);   step θ_se,θ_sd,θ_emb,θ_pred
3.  (ii)  L_cip  ← MSE(cipher recon); step θ_kp,θ_enc,θ_dec
4.  (iii) forward_full → L_B, L_E
          L_joint ← L_B + |L_E − λ|;   step θ_AB (all Alice/Bob params)
5.  (iv)  forward_full → L_E;          step θ_E  (Eve does her best)
6.  (v)   [CSI key only] L_key ← agreement loss; step θ_keygen   # §5.2
7.  λ ← Λ.step(L_B, L_E)                         # §6
```

Notes:
- In step (iii), Eve's sub-network is kept in the graph so the gradient of
  $\lvert L_E-\lambda\rvert$ flows back to Alice (she learns to confuse Eve); Eve's
  *parameters* are **not** in `opt_joint`, so they are not updated there.
- Step (iv) re-runs the forward and updates **only** Eve, realising the adversary.
- Gradient norm clipping (`grad_clip`) is applied per sub-step.
- The same key bundle is reused across sub-steps within a batch.

---

## 5. Improvement A — CSI-based neural key generation

Replaces the externally-shared random key with one **derived from the reciprocal
physical channel**. Code: `securedsc/channels.py` (`ReciprocalChannel`),
`securedsc/keygen.py` (`CSIKeyGenerator`).

### 5.1 Reciprocal CSI model

A ground-truth Alice–Bob channel realisation $\mathbf g\sim\mathcal N(0,I_d)$ is
drawn per sample. The two reciprocal views and the parties' noisy observations are
$$
\mathbf h_{AB}=\rho\,\mathbf g+\sqrt{1-\rho^2}\,\mathbf e_A,\quad
\mathbf h_{BA}=\rho\,\mathbf g+\sqrt{1-\rho^2}\,\mathbf e_B,
$$
$$
\mathbf c_A=\mathbf h_{AB}+\sigma_{\text{est}}\boldsymbol\eta_A,\qquad
\mathbf c_B=\mathbf h_{BA}+\sigma_{\text{est}}\boldsymbol\eta_B,
$$
with $\mathbf e,\boldsymbol\eta\sim\mathcal N(0,I)$ independent and
$\sigma_{\text{est}}$ from the CSI estimation SNR. Consequently
$\text{corr}(\mathbf c_A,\mathbf c_B)\!\approx\!\rho^2$ (reciprocal). **Eve's**
observations $\mathbf c_{E}$ use a *fresh independent* $\mathbf g$, so her CSI is
uncorrelated with the legitimate link. *(Verified in `tests/test_reciprocal.py`:
Alice–Bob correlation $>0.7$ at $\rho=0.95$, Alice–Eve $\approx 0$, and monotone
in $\rho$.)*

### 5.2 Neural key generator + reconciliation

A shared MLP $G_\phi$ (`CSIKeyGenerator`) maps a party's CSI to **soft bits**
$\mathbf s=\tanh(G_\phi(\mathbf c))\in(-1,1)^{B\cdot r}$, where $r$ = `recon_repeat`.
Hard bits are $\text{sign}(\mathbf s)$.

**Reconciliation (repetition code).** Group the $B\cdot r$ hard bits into $B$
blocks of $r$ and take a majority vote per block → $B$ reconciled bits
(`majority_reconcile`). Larger $r$ lowers the Alice/Bob mismatch at the cost of
more CSI bandwidth *(verified in `tests/test_keygen.py`)*. The cipher path uses an
**identical** reconciled key (after the public reconciliation exchange Bob adopts
Alice's bits); the reported agreement is the **raw** pre-reconciliation quality —
the meaningful security signal.

**Training objective** (step (v)): pull Alice/Bob soft bits together, push Eve's
away,
$$
L_{\text{key}}=\big\lVert \mathbf s_A-\mathbf s_B\big\rVert^2
+\big[\,\mu-\lVert \mathbf s_E-\text{sg}[\mathbf s_A]\rVert^2\,\big]_+,
$$
margin $\mu$ = 0.5. Because Eve's CSI is independent, the second term is mostly
satisfied by construction; the first term sharpens Alice/Bob agreement.

The reconciled key replaces $\mathbf k$ into $g_{\text{kp}}$; the random baseline
(`RandomKeyGenerator`) instead shares an identical random $\mathbf k$ for
Alice/Bob and draws an independent one for Eve.

---

## 6. Improvement B — adaptive secrecy-weight scheduler

Code: `securedsc/lambda_sched.py`.

The fixed hyperparameter $\lambda$ is hard to set because the right value scales
with $\ln V$ (corpus-dependent), and $\lvert L_E-\lambda\rvert$ is a *two-sided*
pull: too-small $\lambda$ actively trains Alice to *help* Eve. We replace it with
a feedback controller targeting a desired Eve loss
$\tau=\text{target\_frac}\cdot\ln V$ (near random-guessing):
$$
\widehat{L}_E^{(t)}=\beta\,\widehat{L}_E^{(t-1)}+(1-\beta)L_E^{(t)},\qquad
\lambda^{(t+1)}=\text{clip}\!\big(\lambda^{(t)}+\eta\,(\tau-\widehat{L}_E^{(t)}),\,
[\,\lambda_{\min},\lambda_{\max}\,]\big),
$$
with EMA factor $\beta$ = `ema`, step $\eta$ = `eta`, and
$\lambda_{\max}=\ln V$ by default. If Eve is doing *better* than the target
($\widehat L_E<\tau$), $\lambda$ rises to push harder; if she is already worse,
$\lambda$ relaxes. `FixedLambda` returns a constant (with a runtime warning if it
is set below $0.5\ln V$). *(Update rule and bounds verified in
`tests/test_lambda.py`.)*

This removes manual tuning (one specifies a *difficulty fraction*, not an absolute
weight) and stabilises training across seeds (§8).

---

## 7. Evaluation protocol

Code: `securedsc/eval.py`, driven by `scripts/run_eval.py` and
`scripts/run_ablation.py`.

### 7.1 Metrics

- **BLEU (1–4 gram)** — self-contained implementation (`securedsc/metrics.py`):
  modified $n$-gram precision $p_n=\frac{\sum \min(\text{cand},\text{ref})}{\#\,n\text{-grams}}$,
  geometric mean over $n=1..N$ with $N$ **capped at the candidate length**, NLTK
  "method-1" smoothing (zero precisions → $\epsilon/\text{total}$), and brevity
  penalty $\text{BP}=\min(1,e^{1-r/c})$ for ref/cand lengths $r,c$. Reported for
  Bob and Eve. *(Edge cases unit-tested in `tests/test_bleu.py`.)*
- **Key agreement rate** $\tfrac1B\sum\mathbb 1[\text{sign}(k_{A,i})=\text{sign}(k_{B,i})]$
  and **Eve mismatch** $1-\text{agreement}(k_E,k_A)$.
- **Training stability**: seed-to-seed mean ± std of final BLEU and the Bob–Eve
  gap; convergence of $L_B,L_E,\lambda$ curves.

### 7.2 Sweeps and the ablation grid

1. **BLEU vs SNR** over `eval.snr_db_list`, for **AWGN and Rayleigh** — Bob and Eve
   curves (`bleu_vs_snr`). Sanity target (paper): BLEU(Bob) → ~0.9+, BLEU(Eve) ≲ 0.2.
2. **Key quality vs CSI correlation** over `eval.rho_list` — Alice/Bob agreement and
   Eve mismatch vs $\rho$ (`key_metrics_vs_correlation`); CSI keys only.
3. **Ablation grid** $\{\text{random key, CSI key}\}\times\{\text{fixed }\lambda,
   \text{adaptive }\lambda\}$, multi-seed (`scripts/run_ablation.py`), reporting
   mean ± std of BLEU(Bob), BLEU(Eve), gap, agreement, mismatch, plus training-curve
   and BLEU-vs-SNR comparison plots and a CSV/JSON/Markdown table.

---

## 8. Experimental setup

| Component | Default | Config key |
|---|---|---|
| Dataset | Europarl v7 English (cleaned: lowercase, punctuation stripped) | `data.source=europarl` |
| Sentence length | $L=30$ (pad/`<start>`/`<end>`) | `data.max_len` |
| Vocabulary | most-frequent, cap $V=10^4$, `<pad><start><end><unk>` | `data.max_vocab` |
| $d,\,\text{heads},\,\text{ff}$ | 128, 8, 512 | `model.*` |
| Depth | 3 semantic / 2 cipher layers | `model.num_*_layers` |
| Channel symbols | $C=16$ per token | `model.channel_dim` |
| Key | $B=64$ bits, $T=4$ key tokens, $r=3$ repeat | `model.key_len,n_key_tokens`, `keygen.recon_repeat` |
| Channel | AWGN/Rayleigh, train SNR 12 dB | `channel.*` |
| CSI | $\dim=32$, $\rho=0.95$, est-SNR 15 dB | `channel.csi_*` |
| $\lambda$ | fixed $\approx 0.9\ln V$ (8.0) or adaptive ($\tau=0.9\ln V$) | `lam.*` |
| Optimiser | Adam; lr 1e-4 (joint), 1e-3 (sub/Eve/key); clip 1.0 | `train.*` |
| Schedule | 40 epochs, batch 128 | `train.epochs,batch_size` |
| Seeds | 0,1,2 (ablation) | `--seeds` |

Everything is YAML-driven and seeded (Python/NumPy/PyTorch) for reproducibility;
configs and the full `Config` are saved into each checkpoint. A tiny CPU
`configs/smoke.yaml` runs the entire pipeline in seconds for CI.

---

## 9. Results tables to populate

Fill from `results/ablation/ablation_table.csv` and the eval JSONs.

**Table 1 — Ablation (final BLEU, mean ± std over seeds; train SNR 12 dB, AWGN).**

| Key | $\lambda$ | BLEU(Bob) ↑ | BLEU(Eve) ↓ | Bob−Eve gap ↑ | A/B agreement ↑ |
|---|---|---|---|---|---|
| random | fixed | | | | 1.00 |
| random | adaptive | | | | 1.00 |
| CSI | fixed | | | | |
| CSI | adaptive | | | | |

**Table 2 — BLEU vs SNR** (Bob/Eve, AWGN and Rayleigh): per-SNR rows from
`eval.json["bleu_vs_snr_{awgn,rayleigh}"]`.

**Table 3 — Key quality vs $\rho$**: agreement / Eve-mismatch from
`eval.json["key_vs_correlation"]`.

**Figures**: `bleu_vs_snr.png`, `key_vs_correlation.png`,
`ablation/training_curves.png`, `ablation/ablation_bleu_vs_snr.png`.

---

## 10. Code map (equation → file)

| Item | File:symbol |
|---|---|
| Configs / YAML | `securedsc/config.py` |
| Data, vocab, cleaning, toy corpus | `securedsc/data.py` |
| Channels, power norm, reciprocal CSI | `securedsc/channels.py` |
| Key generators, reconciliation, $L_{\text{key}}$ | `securedsc/keygen.py` |
| $\lambda$ schedulers | `securedsc/lambda_sched.py` |
| NN blocks (pos-enc, cross-attn cipher, channel codec, key-proc) | `securedsc/modules.py` |
| Model, 3 forward paths, param groups | `securedsc/model.py` |
| BLEU, key agreement | `securedsc/metrics.py` |
| Algorithm 1 trainer | `securedsc/train.py` |
| Sweeps + plots | `securedsc/eval.py` |
| Train / eval / ablation / data download | `scripts/*.py` |
| Unit tests (channel, CSI corr, key, $\lambda$, BLEU) | `tests/*.py` |

---

## 11. Design decisions and deviations from the paper

- **Cipher via cross-attention to key tokens.** The encryptor/decryptor are
  Transformer *decoder* stacks that cross-attend to key tokens, giving a concrete
  correct-key-inverts / wrong-key-fails mechanism. The paper describes
  encryptor/decryptor abstractly; this is our faithful instantiation.
- **Non-autoregressive decoding** (whole-sentence prediction), as in DeepSC, for
  efficiency and stability.
- **Separate optimisers per sub-step.** Faithful to the alternating schedule of
  Algorithm 1; parameters shared between a sub-step group and the joint group
  carry independent Adam moments (a standard alternating-training pattern).
- **$L_{\text{sem}}$ as a semantic autoencoder** and **$L_{\text{cip}}$ as feature
  MSE** isolate the two codecs, as intended by the four-way schedule.
- **Mutual-information regulariser** from the original DeepSC is omitted; CE is
  used for the semantic objective (common simplification).

---

## 12. Threats to validity / limitations

- **Dataset entropy is decisive.** On a tiny corpus (e.g. the 20-sentence toy set)
  Eve can *memorise* all messages and BLEU(Eve)≈BLEU(Bob) regardless of the cipher;
  secrecy claims require a high-entropy corpus (Europarl). This is the dominant
  experimental pitfall.
- **$\lambda$ sign/scale.** Because $\lvert L_E-\lambda\rvert$ is two-sided, an
  under-set $\lambda$ degrades secrecy; the adaptive scheduler and a runtime
  warning mitigate this.
- **Reconciliation is modelled, not cryptographic.** Repetition-majority plus a
  public exchange yields an identical key but does not provide formal key-secrecy
  guarantees; security is argued empirically via Eve's BLEU and key mismatch.
- **Rayleigh equalisation** is learned (no explicit CSI to the data path); results
  may differ from a coherent-detection baseline.
- **CSI generator is a synthetic reciprocity model**, parameterised by $\rho$ and
  estimation SNR; real measured CSI may exhibit different statistics.

---

## 13. Reproducibility checklist

- Pinned `requirements.txt` (torch 2.12, verified Python 3.11–3.13).
- Deterministic seeding of Python/NumPy/PyTorch (`utils.set_seed`).
- Every run saves model + key-gen + full config + per-epoch history
  (`results/<run>/ckpt.pt`, `history.json`).
- One-command dataset prep (`scripts.get_europarl`), training, evaluation, and the
  full ablation grid.
- Unit tests assert the core mathematical properties (channel noise vs SNR,
  reciprocal-CSI correlation, key reconciliation, $\lambda$ feedback, BLEU).

---

## 14. References

1. Z. Shi *et al.*, "Secure Transmission in Wireless Semantic Communications With
   Adversarial Training," *IEEE Communications Letters*, 2025.
2. H. Xie, Z. Qin, G. Y. Li, B.-H. Juang, "Deep Learning Enabled Semantic
   Communication Systems" (DeepSC), *IEEE Trans. Signal Processing*, 2021.
3. A. Vaswani *et al.*, "Attention Is All You Need," *NeurIPS*, 2017.
4. K. Papineni *et al.*, "BLEU: a Method for Automatic Evaluation of Machine
   Translation," *ACL*, 2002.
5. Physical-layer key generation from reciprocal channel/CSI (survey), for the
   reciprocity and reconciliation background of Improvement A.

> Replace citation details with the exact bibliographic entries for your venue.
