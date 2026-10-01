# Resume Bullet Points & Technical Interview Guide: HOMERs

This guide provides battle-tested, high-impact resume bullet points, portfolio blurbs, and technical interview talking points tailored for **HOMERs**.

---

## 1. Ready-to-Copy Resume Bullet Points (Google / Meta / Quant XYZ Format)

Use the section that matches the role you are targeting. Each bullet is written in the gold-standard **"Accomplished [X], as measured by [Y], by doing [Z]"** format.

### Option A: Machine Learning Engineer (MLE) / Applied AI Researcher

* **Architected and deployed an end-to-end sports intelligence platform** modeling NBA possession dynamics across 10 seasons (13,000+ games, 2.5M+ plays) using an autoregressive causal transformer (Tempo) and a multi-task neural shot engine (ShotNet).
* **Trained a 4-layer decoder-only transformer with causal scaled dot-product attention** over a discrete 31-token game grammar, conditioning on non-linear decaying game state $\left(\frac{\Delta\text{score}}{\sqrt{t+1}}\right)$ and achieving **42.1% next-event accuracy (4.44 perplexity)** and monotonic win-probability calibration ($R^2 > 0.99$).
* **Formulated a hierarchical Bayesian multi-task deep neural network (ShotNet)** to predict 8 shot zones, 9 physical shot styles, and conditional make probability, leveraging permutation-invariant set pooling for 10 on-court players and coaching latent vectors; **reduced shot-type cross-entropy loss by 7.6% (1.694 vs 1.833 league baseline)** on 233,000+ out-of-time held-out attempts.
* **Engineered a client-side neural inference runtime in pure vectorized JavaScript/Wasm**, compiling trained PyTorch weights into a zero-server static dashboard delivering **<1.8ms forward-pass latency** with strict numerical parity to PyTorch FP32 ($\Delta < 10^{-4}$).
* **Designed a 10,000-path Monte Carlo game rollout engine** simulating remaining possession trajectories from any clutch game state, calculating real-time Possession Leverage Index (LI) and counterfactual tactical scheme impact in **<45ms**.

---

### Option B: Quantitative Researcher / Sports Data Scientist

* **Engineered an econometric and deep learning sports prediction engine** analyzing 2.5M+ possessions across 10 NBA seasons, establishing strict temporal out-of-time splits (2016–2025 train vs 2025–26 test) with zero forward data leakage.
* **Developed a low-rank bilinear matrix factorization model (Poisson & Binomial GLMs)** with exposure modeling on partial possessions to quantify latent offensive/defensive player interaction vectors $U_X \cdot V_Y$ and isolate individual defensive deterrence.
* **Built a multi-resolution feature pipeline** integrating Second Spectrum spatial tracking, Synergy Sports tactical play types, on-court lineup states, and referee/coaching indicators to isolate possession-level Expected Possession Value (EPV) and Leverage Index (LI).
* **Identified and resolved critical temporal anomalies in official NBA data feeds**, designing a deterministic multi-pass state machine that re-ordered **17,890 out-of-sequence events across 4,235 games**, driving a **9.4% reduction in sequence model perplexity (4.90 to 4.44)**.
* **Evaluated probabilistic forecasts using quarter-by-quarter Brier score decomposition** (achieving 0.0868 in Q4 clutch scenarios) and reliability curves across probability deciles, outperforming historical score-and-clock logistic baselines.

---

### Option C: ML Systems / Full-Stack AI Engineer

* **Engineered a high-throughput, fault-tolerant ingestion ETL pipeline** with automated exponential backoff and SQLite/Parquet caching to process multi-gigabyte datasets from NBA Stats API and Basketball-Reference.
* **Developed a deterministic 5-on-5 lineup tracking state machine** reconstructing complete on-court player rosters across **97%+ of 2.5M historical possessions** from fragmented substitution logs.
* **Eliminated server-side infrastructure and cloud GPU costs entirely** by designing a zero-server interactive single-page dashboard (`dashboard.html`) bundling pre-computed vector indexes and a client-side linear algebra forward-pass engine.
* **Built interactive real-time visual analytics components** including kernel-density-smoothed hexbin shot charts, counterfactual matchup simulators, broadcast replay scrubbers, and coach tactical radar matrices in canvas/SVG.
* **Containerized and automated testing/deployment pipelines** with bash/CLI harnesses (`./run.sh`), enabling offline synthetic data generation, automated regression tests, and single-click production compilation.

---

## 2. Project Portfolio & Elevator Pitches

### 1-Sentence Summary (for Resume Header or LinkedIn Project Description)
> **HOMERs:** An end-to-end deep learning framework and zero-server interactive intelligence engine modeling NBA possession dynamics, shot selection, and counterfactual game simulations across 10 seasons (13k+ games, 2.5M+ plays).

### 3-Sentence Technical Summary (for Portfolio Website or GitHub Bio)
> Built an end-to-end sports machine learning system featuring an autoregressive causal transformer for next-play generation and continuous win probability, alongside a hierarchical multi-task neural network predicting shot spatial zones, styles, and make probabilities conditioned on 10-player on-court gravity and coaching schemes. Solved large-scale tracking data anomalies by reconstructing 97%+ of 5-on-5 lineups and correcting 17,890+ out-of-order events. Deployed models into an ultra-low-latency, zero-server client runtime delivering sub-2ms in-browser neural inference and 10,000-path Monte Carlo clutch rollouts.

---

## 3. Technical Interview Cheat Sheet (Deep-Dive Talking Points)

### Q1: "Walk me through the system architecture of your project."
* **Answer Blueprint:**
  1. **Data Ingestion & State Machine:** Ingested 10 seasons (~13k games) from NBA Stats API (`PlayByPlayV3`, `BoxScoreMatchupsV3`, Synergy, Second Spectrum) into Parquet format. Built a state machine to reconstruct 5-on-5 on-court lineups from substitution logs (97% coverage) and re-ordered 17,890 out-of-sequence events.
  2. **Model 1 (Tempo):** 4-layer causal decoder-only transformer with scaled dot-product attention (FlashAttention/SDPA) over a 31-token play vocabulary. Input embeds tokens, position, and continuous game dynamics including a continuous lead-decay term $\Delta\text{score}/\sqrt{t+1}$. Multi-task loss: Next-play cross-entropy + Home-win binary cross-entropy.
  3. **Model 2 (ShotNet):** Multi-task deep neural network predicting shot zone (8 classes), shot style (9 classes), and make probability. Uses hierarchical Bayesian priors where the network learns residual shifts away from empirical player priors, conditioned on 10-player permutation-invariant set embeddings and coaching schemes.
  4. **Model 3 (Matchup Engine):** Bilinear Poisson/Binomial GLM with exposure modeling on partial possessions, isolating latent player interaction factors ($U_X \cdot V_Y$).
  5. **Client-Side Runtime & Simulation:** Compiled model weights into a zero-server single-file dashboard (`dashboard.html`) executing vectorized linear algebra in pure JS with sub-2ms forward-pass latency and a 10,000-path Monte Carlo rollout simulator.

---

### Q2: "What was the most challenging data engineering or modeling problem you faced?"
* **Answer Blueprint (The Chronological Out-of-Order Logging Bug):**
  * *Context:* "When training the sequence transformer on raw NBA API play-by-play, validation perplexity plateaued around 4.90."
  * *Investigation:* "I inspected play sequence anomalies and discovered that official scorers often log plays late (e.g. substitutions, technical fouls, or reviewed plays) by assigning high `actionNumber`s that don't match game-clock time. Sorting solely by `actionNumber` caused 17,890 plays across 4,235 games to be displaced by several minutes in the causal sequence."
  * *Resolution:* "I developed a deterministic chronological state reconciliation algorithm in `models/tokenize_pbp.py` that sorts primarily by period, secondarily by remaining game clock, and tertiarily by event dependency rules."
  * *Impact:* "Resolving this restored true causal ordering, immediately dropping sequence perplexity from 4.90 to 4.44 and boosting next-play accuracy to 42.1%."

---

### Q3: "Why use hierarchical Bayesian priors in ShotNet instead of a pure end-to-end neural network?"
* **Answer Blueprint:**
  * "In sports data, sample sizes follow an extreme power law: a star like Luka Dončić takes 1,500 shots per season, while role players or rookies take fewer than 50. A pure end-to-end neural net severely overfits on high-volume shooters and generates erratic predictions on low-volume players."
  * "To solve this, I formulated ShotNet as a **residual prior architecture**:
    $$\text{logits}_{\text{zone}} = \text{Prior}_{\text{zone}}[i] + W_{\text{zone}} \cdot h(\text{context})$$
    The network is initialized with zero weights on the contextual heads, starting exactly at the Dirichlet/Beta-smoothed historical baseline of the shooter. The neural layers learn solely the contextual *perturbation* (the tactical effect of the defender, the 5-man floor spacing, and the coach's defensive scheme)."
  * "This guarantees that the model never performs worse than the player's historical baseline, while gracefully capturing defensive deterrence effects."

---

### Q4: "How does the in-browser neural runtime achieve PyTorch parity without a backend server?"
* **Answer Blueprint:**
  * "Instead of hosting a costly Python/Flask/FastAPI microservice with GPU instances, I serialized the trained model parameters (weight matrices, biases, player lookup tables) into compact Float32 arrays embedded directly inside `dashboard.html`."
  * "I authored a lightweight, vectorized matrix math library in JavaScript executing dense GEMM (General Matrix Multiply), ReLU/GELU activations, layer normalization, and Softmax."
  * "Benchmarking showed the forward pass runs in under 1.8ms on consumer CPU hardware, maintaining numerical precision parity with PyTorch FP32 to within $\Delta < 10^{-4}$."
  * "This enables zero-latency interactive simulations where users can drag-and-drop matchups, adjust clocks, and run 10,000 Monte Carlo rollouts offline with zero server costs."

---

## 4. Key Metrics & Technical Terms for Your Resume / LinkedIn

| Category | Keywords & Metrics to Highlight |
|---|---|
| **Architectures** | Autoregressive Transformer, Causal Self-Attention, Multi-Task Learning, Permutation-Invariant Set Representations, Hierarchical Bayesian Priors, Bilinear GLM |
| **Scale & Data** | 10 NBA Seasons (2016–2026), 13,000+ Games, 2.5M+ Possessions, 233,000+ Field Goal Attempts, Parquet, PyArrow, SQLite |
| **Performance Metrics** | 42.1% Next-Event Accuracy, 4.44 Perplexity, 1.694 Shot-Type Log Loss (-7.6% vs League), 0.0868 Q4 Brier Score, <1.8ms Client Latency |
| **Engineering & Infra** | PyTorch 2.x, FlashAttention/SDPA, Pure JS/Wasm Linear Algebra Runtime, Monte Carlo Rollout Engine, Out-of-Time Temporal Cross-Validation |
| **Domain Analytics** | Expected Possession Value (EPV), Possession Leverage Index (LI), Second Spectrum Tracking, Synergy Sports Play Types, Defensive Deterrence Modeling |
