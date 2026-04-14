# TriAttention theory brief (Round 1, tasks T1–T2)

This note extracts the paper’s technical design at the theory/method level, without digging into vLLM/runtime implementation details. The focus is the paper’s motivation, pre-RoPE concentration observation, RoPE-to-trigonometric derivation, exact scoring equations, future-offset handling, GQA aggregation, ablations, and remaining uncertainties. Inline citations use numbered sources listed below.

## Questions answered

1. **What motivates TriAttention?**  
   The paper argues that most prior KV-cache compression methods estimate token importance from **recent post-RoPE queries**, but RoPE rotates query directions with position. Because of that rotation, only a very small recent window has the “right” orientation to predict future attention, so important keys can look unimportant during the short observation window and get evicted too early. The paper frames this as especially harmful for long reasoning and retrieval-like heads, where a token may stay dormant for a long time and then become critical later [S1][S2].

2. **What is the core empirical observation?**  
   In **pre-RoPE space**, both Q and K vectors are reported to cluster around **stable, non-zero centers** across positions and across different input sequences. The paper names this phenomenon **Q/K concentration**. RoPE then rotates these concentrated vectors into arcs in post-RoPE space, which is why the post-RoPE view looks unstable while the pre-RoPE view looks stable [S1][S2].

3. **How is concentration measured?**  
   The main metric is **Mean Resultant Length (MRL)**:
   
   \[
   R = \frac{\|\mathbb{E}[q]\|}{\mathbb{E}[\|q\|]}
   \]
   
   with the per-frequency-band version
   
   \[
   R_f = \frac{\|\mathbb{E}[q_f]\|}{\mathbb{E}[\|q_f\|]}.
   \]
   
   Intuition: \(R\to1\) means vectors align tightly around a common direction; \(R\to0\) means directions are dispersed. The paper visualizes concentration on each head’s **dominant frequency bands**, chosen by expected contribution
   
   \[
   C_f = \mathbb{E}[\|q_f\|] \cdot \mathbb{E}[\|k_f\|],
   \]
   
   typically keeping the top 2 bands for plots [S1][S2].

4. **How do they derive a trigonometric series from RoPE attention?**  
   The derivation is:
   - represent each 2D RoPE band as a complex scalar,
   - write RoPE as a complex rotation \(\tilde q_f(p)=q_f e^{i\omega_f p}\), \(\tilde k_f(p)=k_f e^{i\omega_f p}\),
   - take the real part of the rotated query-key product,
   - obtain a distance-only logit form:
     
     \[
     \text{logit}(q,k)=\sum_f \|q_f\|\,\|k_f\|\cos(\omega_f\Delta + \phi_f),
     \]
     
     where \(\Delta=p_q-p_k\) and \(\phi_f=\arg(q_f)-\arg(k_f)\).
   - if pre-RoPE Q and K are approximately constant across tokens (or well-approximated by their centers), then the coefficients become approximately fixed, so the logit becomes
     
     \[
     \sum_f [a_f\cos(\omega_f\Delta)+b_f\sin(\omega_f\Delta)],
     \]
     
     i.e. a **trigonometric series over query-key distance** [S1][S2].

5. **What assumptions/approximations make the derivation useful for pruning?**  
   The paper relies on several approximations/design assumptions [S1][S2]:
   - **Concentration assumption:** pre-RoPE \(q_f\) and \(k_f\) are close to stable centers.
   - **Center substitution:** for analysis/reconstruction, replace token-specific \(q_f,k_f\) with \(\mathbb{E}[q_f],\mathbb{E}[k_f]\).
   - **Future-query proxy:** at inference time, future queries are approximated using the **Q center** \(\mathbb{E}[q_f]\).
   - **Logit-level approximation:** the model predicts relative attention tendencies using pre-softmax logits, not a formal end-to-end softmax/value-output model.
   - **Distance-preference prior:** future importance is treated as depending strongly on distance structure, then corrected with a norm-based term when concentration is weaker.

6. **What is the final scoring function?**  
   The paper’s final per-key score has two pieces [S1][S2]:

   **(a) Trigonometric score**
   \[
   S_{\mathrm{trig}}(k,\Delta)=\sum_f \|\mathbb{E}[q_f]\|\,\|k_f\|\cos(\omega_f\Delta+\phi_f),
   \]
   where \(\phi_f = \arg(\mathbb{E}[q_f]) - \arg(k_f)\).

   **(b) Norm-based residual score**
   \[
   S^{(0)}_{\mathrm{norm}}(k)=\sum_f \mathbb{E}[\|q_f\|]\,\|k_f\|,
   \]
   then weighted by concentration:
   \[
   S_{\mathrm{norm}}(k)=\sum_f (1-R_f)\,\mathbb{E}[\|q_f\|] \,\|k_f\|
   \]
   and equivalently
   \[
   S_{\mathrm{norm}}(k)=\sum_f \big(\mathbb{E}[\|q_f\|]-\|\mathbb{E}[q_f]\|\big)\,\|k_f\|.
   \]

   **(c) Combined score**
   \[
   S(k,\Delta)=S_{\mathrm{trig}}(k,\Delta)+S_{\mathrm{norm}}(k).
   \]

   Intuitively, the trig term models each head’s learned distance preference, while the norm term preserves a fallback signal for the minority of less-concentrated heads [S1][S2].

7. **How is future-offset averaging defined?**  
   Because a cached key may be queried from many later positions, the paper averages the score over a geometric set of future offsets:
   
   \[
   \tilde S(k)=\frac{1}{|\mathcal D|}\sum_{\delta\in\mathcal D} S(k,\Delta+\delta),
   \]
   
   with
   
   \[
   \mathcal D = \{1,2,4,\dots,2^{16}\}.
   \]
   
   So the production scoring rule is not “current distance only”; it is an average over a broad future-distance prior. Appendix G reports that using future offsets helps, and that **geometric spacing** is much better than linear spacing [S1][S2].

8. **How is GQA handled?**  
   For Grouped-Query Attention, one KV head is shared by multiple query heads, so each key gets multiple query-head-specific scores \(\tilde S^{(g)}(k)\). The paper says raw scales differ across heads, so it uses **normalize then aggregate** [S1][S2]:
   
   \[
   \hat S^{(g)}(k)=\frac{\tilde S^{(g)}(k)-\mu_g}{\sigma_g}
   \]
   
   and then
   
   \[
   S_{\mathrm{final}}(k)=\max_g \hat S^{(g)}(k).
   \]
   
   Interpretation: a key is kept if **any** associated query head strongly prefers it [S1][S2].

9. **What pruning schedule is described?**  
   The paper prunes in **windows of 128 generated tokens**. Every \(\beta=128\) tokens, if the cache exceeds budget \(B\), it scores all keys, retains the top-\(B\), and evicts the rest. This is explicitly a compute-overhead tradeoff rather than per-token rescoring [S1][S2].

10. **What do the main ablations say?**  
    Key ablation findings reported in the paper [S2]:
    - **Remove trig score:** AIME24 drops from 42.1 to 18.8 and AIME25 from 32.9 to 21.2 (Table 3A). This is the strongest evidence that the distance-preference term is the main driver.
    - **Remove concentration weighting \(R\):** AIME24 drops 42.1 → 41.3 and AIME25 32.9 → 28.7 (Table 3B), showing the adaptive weighting matters, especially on AIME25.
    - **Cross-domain calibration:** coding-calibrated stats still work reasonably on reasoning (44.2/29.2 vs 42.1/32.9 on AIME24/25; Table 3C), supporting the claim that concentration is largely model-intrinsic.
    - **Future offsets:** max distance 4096 performs better than 128 (48.8 vs 41.7 on AIME24), and **geometric spacing** crushes linear spacing (45.8 vs 28.7; Table E).
    - **Calibration sensitivity:** 50k, 200k, and 960k token calibration sets are close (45.4, 45.8, 45.8 on AIME24); data quality is also not tightly correlated with accuracy (Table F).
    - **Architecture generality:** MLA still shows strong concentration; e.g. 96.6% of MLA heads have \(R>0.95\) versus 84.7% for Qwen3 GQA (Table G).

11. **What explicit limits/uncertainties does the paper itself acknowledge?**  
    The appendix states that current speedups could be improved with a **dedicated hardware-aware kernel**, and future work includes broader evaluation on **coding and agentic tasks** plus potentially **head-specific budgets** [S2]. The paper also implicitly acknowledges that concentration is not perfect, because it keeps a norm-based complement and reports reconstruction correlations with means above 0.5 rather than near 1.0 [S1][S2].

## Numbered sources

1. **[S1] arXiv HTML (main paper text + appendix links)** — *TriAttention: Efficient Long Reasoning with Trigonometric KV Compression*.  
   URL: https://arxiv.org/html/2604.04921v1

2. **[S2] arXiv PDF** — *TriAttention: Efficient Long Reasoning with Trigonometric KV Compression*.  
   URL: https://arxiv.org/pdf/2604.04921.pdf

3. **[S3] Project page** — TriAttention overview and headline results.  
   URL: https://weianmao.github.io/tri-attention-project-page/

4. **[S4] Official repository** — README and public project framing.  
   URL: https://github.com/WeianMao/triattention

## Evidence table

| Topic | Direct evidence | What it supports | Notes |
|---|---|---|---|
| Motivation against post-RoPE observation windows | Intro + §2.3 explain that post-RoPE queries rotate with position, so only a tiny recent window remains representative; larger windows do not keep helping [S1][S2] | Why the authors reject recent-attention-based eviction as unstable for long reasoning | Core motivation is theoretical plus literature-backed, not a new standalone benchmark in the main text |
| Pre-RoPE Q/K concentration | Observation 3.1, Figure 2, and Appendix I show stable non-zero centers across positions, contexts, and architectures [S1][S2] | Basis for replacing token-specific Q/K with centers | Strong empirical observation; not a formal theorem |
| Measurement of concentration | MRL definition in §3.1 and Appendix B.5: \(R=\|E[q]\|/E\|q\|\), per-band \(R_f\); dominant bands chosen by \(C_f=E\|q_f\|E\|k_f\|\) [S1][S2] | Exact statistic used for quantification and visualization | Design weighting later uses query-side \(R_f\) |
| RoPE-to-trig derivation | §3.2 + Appendix B.1–B.4 derive \(\sum_f |q_f||k_f|\cos(\omega_f\Delta+\phi_f)\), then fixed coefficients when Q/K are constant/centered [S1][S2] | The method’s distance-preference model | Frequencies are geometric, so “trigonometric series” is Fourier-like but not classical Fourier |
| Reconstruction validation | §3.3 + Appendix B.6 define reconstruction correlation \(\bar r\) over log-spaced distances; example head gets 0.72; cross-model distributions have means >0.5 [S1][S2] | Evidence that center-derived curves track real attention patterns | Supports usefulness, not exact recovery |
| Final trig score | Eq. 6 uses the Q center \(E[q_f]\) and the **actual cached key** \(k_f\) [S1][S2] | Exact practical scoring form | Important: inference does **not** replace the key by its center |
| Norm complement and adaptive weighting | Eqs. 7–10 define norm score and concentration-based weighting by \((1-R_f)\) [S1][S2] | Exact final score construction | Suggests norm term is a residual correction for weakly concentrated bands |
| Future-offset averaging | Eq. 11 with \(\mathcal D=\{1,2,4,\dots,2^{16}\}\); Appendix G ablates range/spacing [S1][S2] | How the paper handles unknown future query positions | Important practical prior over future distances |
| GQA aggregation | Eqs. 12–13: per-head z-score then max over query heads [S1][S2] | Exact multi-query-head aggregation | Max favors recall: keep if any query head wants it |
| Ablation strength | Table 3A/3B/3C, Table E, Table F, Table G [S2] | Which pieces matter most empirically | Strongest signal is removing \(S_{trig}\) |
| Explicit paper limitations | Appendix A calls for dedicated kernels, broader task coverage, and possibly head-specific budgets [S2] | What the paper itself does **not** claim to have solved | Good boundary for later implementation research |

## Observations vs inferences

### Observations directly supported by the paper

- **Post-RoPE recent-query methods are argued to be unstable** because RoPE rotates query orientation with position, shrinking the representative observation window [S1][S2].
- **Pre-RoPE Q/K concentration exists empirically** across most heads and remains stable across positions and content domains [S1][S2].
- **MRL is the paper’s concentration metric**, and the paper uses per-band \(R_f\) in the method itself [S1][S2].
- **RoPE attention can be written as a sum of cosine terms over relative distance**, and under concentration/constant-vector assumptions this becomes a trig series with fixed coefficients [S1][S2].
- **TriAttention’s practical score is not just the reconstruction formula**: it uses the Q center plus the actual current key, then adds a concentration-weighted norm term [S1][S2].
- **Future offsets are explicitly averaged**, rather than using only the current query-key distance [S1][S2].
- **GQA uses z-score normalization per query head and max aggregation across heads** [S1][S2].
- **The trig term is empirically essential**; removing it causes the largest ablation drop [S2].

### Inferences / interpretations from this reading

- **TriAttention is best understood as learning a head-specific distance prior from stable pre-RoPE geometry**, then combining that prior with token-specific key magnitude. That framing is consistent with the equations, but the phrase “distance prior” is my interpretation, not the paper’s exact wording.
- **The norm term looks like a residual correction for non-concentrated mass** because Eq. 9 effectively uses \(E\|q_f\|-\|E[q_f]\|\). The paper motivates it as a complement when concentration is weak; calling it a “residual” is interpretive.
- **Future-offset averaging behaves like a hand-crafted prior over where future queries may land**, rather than a learned future-query distribution. The paper does not phrase it this way, but Eq. 11 and Appendix G strongly suggest that role.
- **The GQA max aggregator is recall-oriented**: it preserves a key if any grouped query head thinks the key matters. That is the natural interpretation of Eq. 13, though the paper only states “a key is retained if any query head deems it important” [S1][S2].
- **The method models attention logits more directly than downstream value contribution**. That is important because the final score uses Q/K information, not V norms. This is an interpretation of what the equations emphasize, not an explicit paper claim about superiority over value-aware methods.

## Unresolved gaps

1. **Default future-offset set vs ablation peak is not fully explained.**  
   Eq. 11 defines the default offset set as \(\{1,2,4,\dots,2^{16}\}\), but Table E shows a max distance of **4096** performs better than **65536** on Qwen3-8B/AIME24 (48.8 vs 45.8) [S2]. The paper does not explain why the larger default remains the main definition.

2. **There is an apparent inconsistency in the norm-ablation discussion.**  
   Table 3 reports full TriAttention at **42.1** on AIME24, but the text says removing \(S_{norm}\) drops AIME24 from **45.8** to **40.4** [S2]. That looks like either a typo, a different hidden setting, or a missing table row.

3. **The theory is empirical rather than guaranteed.**  
   The paper shows reconstruction correlations with means above 0.5 and many high-MRL heads, but it does not prove that concentration must hold, nor that center-based scoring will remain reliable for every model/domain/head [S1][S2].

4. **The bridge from logit reconstruction to pruning quality is not formally closed.**  
   The derivation and validation happen at the logit level; the final usefulness for pruning is supported by benchmark results, not by a full theory of post-softmax attention output or value mixing [S1][S2].

5. **Q/K concentration is discussed symmetrically, but adaptive weighting is query-side.**  
   The paper observes both Q and K concentration, yet the adaptive weighting factor uses \(R_f\) computed from query statistics. The paper does not explain why a symmetric Q/K concentration weight would be worse or unnecessary [S1][S2].

6. **The paper explicitly leaves performance engineering unfinished.**  
   Appendix A says more latency reduction likely requires a dedicated hardware-aware kernel; so the current design should be understood as a method-level result, not the final optimized systems form [S2].
