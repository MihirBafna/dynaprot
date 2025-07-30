## Overall Response


**Table 1**: Comparison of **DynaProt** ensemble generation (on ATLAS test set from Jing et al (2024)) against sequence only methods: AFMD (no template), BioEmu, ESM3, ESMDiff .


| Metric                         | **DynaProt**  | AlphaFlow-MD | BioEmu | ESM3 (ID) | ESMDiff (ID) |
|-------------------------------|--------------|---------------|--------|-----------|--------------|
| Global RMSF r (↑)             | **0.71**     |  <u>0.60</u>          |0.63 | 0.19      | 0.49         |
| Per-target RMSF r (↑)         | **0.86**     | <u>0.85</u>          | 0.77|  0.67      | 0.68         |
|Root mean W2-dist Var Contrib (↓)|  **1.18**  | <u>1.30</u>          | 2.04 |  4.35      |  3.37      |
| MD PCA W2 (↓)                 | <u>1.74</u>     |  **1.52**          | 2.05 |2.06      | 2.29         |
| Joint PCA W2 (↓)              | <u>2.39</u>     | **2.25**          | 4.22 | 5.97      | 6.32         |
| Weak contacts J (↑)           | 0.51     |  **0.62**          | 0.33| 0.45      | <u>0.52</u>         |
| Transient contacts J (↑)      | <u>0.29</u>     | **0.41**           | 0.19 | 0.26      | 0.26         |
| # Parameters (↓)              | **2.86M**    | 95M           | 31M| 1.4B      | 1.4B         |
| Ensemble sampling time (↓)    | **~0.14s**   |~6500s        | ~240s | ~70s        | ~70s         |




| Metric                         | **DynaProt**  | ConfDiff-ESM-r3-MD |Str2Str |
|-------------------------------|--------------|----------|---------|
| Pairwise RMSD (=2.89)         | **2.17**         | 3.91     |  |
| RMSF (=0.1.48)                | **1.10**         | 2.79     | |
| Global RMSF r (↑)             | **0.71**     | 0.48     |     |
| Per-target RMSF r (↑)         | **0.86**     | 0.82     |    |
| Root mean W2-dist Var Contrib (↓) | **1.18**  | 1.51     |   |
| MD PCA W2 (↓)                 | 1.74  | **1.66**     |    |
| Joint PCA W2 (↓)              |**2.39**  | 2.89     |   |
| Weak contacts J (↑)           | 0.51         | **0.56**     ||
| Transient contacts J (↑)      | 0.29  |**0.34**     | |
| # Parameters (↓)              | **2.86M**    | -     |   |
| Ensemble sampling time (↓)    | **~0.14s**   | ~570s    | |



**Table 2**:  **DynaProt** zero-shot ensemble generation on Structural Dynamics of BPTI (Shaw et al (2010)). Evaluation metric protocol from Jing et al (2024).
| Metric                           | **DynaProt (BPTI)** |
|----------------------------------|---------------------|
| Pairwise RMSD (=1.57)            | 1.36                |
| RMSF (=0.84)                     | 0.86                |
| Global RMSF r (↑)                | 0.88                |
| Per-target RMSF r (↑)            | 0.88                |
| Root mean W2-dist Var Contrib (↓)| 0.52                |
| MD PCA W2 (↓)                    | 0.49                |
| Joint PCA W2 (↓)                 | 0.81                |
| Weak contacts J (↑)                 |  0.54               |
Transient contacts J(↑)               | 0.54                |
| # Parameters (↓)                 | 2.86M               |
| Ensemble sampling time (↓)       | ~0.05s              |


**Table 3**: **DynaProt** ablations. Replacing Riemannian aware LogFrob loss with canonical MSE, and replacing IPA layers (SE(3) invariance) with MLP blocks. Comparisons on the marginal Gaussian predictions and residue level RMSF Pearson correlation.
| Metric           | DynaProt-M | No LogFrob Loss | No SE(3) Invariance |
|------------------|----------|------------------|----------------------|
| RMWD Var (↓)     | **1.18**     | 2.70             | 1.92                 |
| Sym KL Var (↓)   | **0.91**     | 9.26             | 4.46                |
| RMSF r (↑)       | **0.87**    | 0.38             | 0.48                 |


### Reviewer 1 (JQtJ)
We appreciate the reviewer’s questions regarding the utility of DynaProt’s predictions and whether the improvements in accuracy are meaningful in practice. We believe these improvements are indeed impactful and provide ample explanations below. The first sections (A,B,C) are copied to all reviewers, and an additional reviewer-specific point by point response is provided.

#### (Copied narrative for all reviewers)
#### A. Why the marginal improvements are actually important?

First, while some of the numerical improvements may seem small at first glance, we believe they are quite meaningful coupled with the fact that DynaProt remarkably achieves these results with a model that's just <=2.86M parameters and runs in under a second (0.14s or even 0.05s in some settings, as opposed to AFMD+templates with ~6500s). That makes it extremely fast and lightweight, especially compared to other methods that rely on large-scale pretraining or generative modeling, like almost every other method reliant on AlphaFold2 or ESM (95M, 1.4B params respectively). For structural biologists screening thousands of molecules, traditional methods would take 6500*1000 s ~= 75 days. This would be untenable. So there are two main takeaways:

  - DynaProt's utility as a method: _DynaProt's speed and simplicity make  practical and accessible for everyday use in a structural biology toolkit_ (see Section B below) 
  - DynaProt's perspective for the field: _Perhaps you don’t always need huge models or complex training pipelines to get useful, accurate structural dynamics predictions_
  
  In fact, we hope DynaProt encourages the field to branch out from its current reliance on large generative models—sometimes, carefully chosen and faster architectures can be just as effective, if not more so, for certain scientific tasks.

#### B. Utility of DynaProt's predictions and new experiment of DynaProt's zero-shot cryptic pocket discovery

1. ##### Marginal covariances can be used as a blazing fast replacement to current NMA plugins
    DynaProt predicts marginal covariances—anisotropic Gaussian “blobs” per residue—with high accuracy, outperforming both sequence-based and template-based methods (see updated tables). We apologize if the current version does not motivate the use case of such dynamics readouts, we will make sure to revise the relevant sections of the manuscript and hopefully we can give some insight here. To appreciate the utility of these predictions, it helps to consider how structural biologists currently operate. When analyzing a protein, they often load its structure into PyMOL and use plugin tools that enable Normal Mode Analysis (NMA) to explore dynamics. NMA is also essentially a Gaussian approximation of the protein’s motion and from those collective modes one could study the local residue motion. But, DynaProt offers a significant upgrade as it produces these local anisotropic predictions faster, with much higher accuracy (main text Tables 2,3,4), and crucially, without requiring simulation or long sampling times (AFMD+Templates). This means that, for any static input structure, DynaProt's marginals gives you not just how much a residue is likely to move, but the direction of its motion—and does so much better than NMA. 

2. ##### DynaProt’s zero-shot cryptic pocket discovery
    This directional information can be immediately useful, moreso than invariant descriptors like RMSF. For instance, it can reveal cryptic pockets—transient openings on the protein surface that allow ligands to bind (CITE PocketMiner). These cryptic pockets are often the key to making target proteins druggable. To this end, we applied DynaProt to zero-shot predict the marginal dynamics of the apo form of ADENYLOSUCCINATE SYNTHETASE (PDB ID: 1ade), a protein featured in the PocketMiner paper’s curated set of cryptic pocket examples. When comparing the predicted anisotropic gaussian blobs overlaid with the holo form (1cib), the predicted Gaussian blobs on the residues involved in the pocket remarkably pointed toward the direction of the holo form protein suggesting that it was indeed capturing the cryptic pocket formation. We really wanted to include this as a figure with the Gaussian blobs overlaid, however NeurIPS 2025’s new rules prevent us from attaching new figures/plots. We plan to include it in the final version or supplementary material, as it’s an exciting use case of DynaProt in action: how dynamic readouts could help uncover druggable motions of enzymes with zero additional supervision.


3. ##### DynaProt’s pair residue coupling and ensemble generation
    DynaProt also predicts pairwise residue couplings that indicate how each pair of residues move in a correlated fashion and can generate reasonable structure ensembles much faster than any current method. Despite being significantly smaller in parameter count, DynaProt generates ensembles with higher fidelity than sequence-based methods (see Table 1), outperforms some structure/template-based methods (see Table 2), and is competitive with SOTA approaches like AFMD+Templates. We hypothesize that this could change how researchers interact with structures—imagine loading a structure in PyMOL and instantly getting a plausible ensemble.

4. ##### DynaProt's outputs as future use for design
    Finally, we believe DynaProt's readouts could enable dynamics-conditioned design. For instance, users could specify the anisotropic marginals of regions that should remain rigid or flexible (and with directionality)—e.g., stabilizing a loop or introducing breathing motion near a pocket. Models like ProtComposer (CITE HERE), which condition on geometric ellipsoids to guide secondary structure, could instead condition on DynaProt’s Gaussian ellipsoids to design proteins that exhibit desired dynamic behaviors. The pairwise coupling readouts could also be used to design proteins with coupled movements. We plan to add these ideas in the discussion as future work.


#### Point by point responses
- _Overall, the accuracy improvements seem marginal. It is difficult to assess how relevant small improvements are as the paper does little to illustrate the utility of the predictions for useful scientific tasks. Why are these marginal accuracy improvements important? Please see more in the questions section._

We hope the above sections clarified these concerns about the "marginal improvements" and the utility of the DynaProt predictions.

- _The authors do little to investigate how well their method predicts large scale protein restructuring. Some more detailed insight about systems where the method struggles and performs well would be beneficial._

We’ve added a new experiment on BPTI, a protein with larger-scale dynamics than those in ATLAS, and found that DynaProt models its motions well—demonstrating generalizability beyond small fluctuations, though we expect limitations on extreme restructuring as that is probably too far OOD.

- _One main contribution is going from simple invariant descriptions to include also anisotropic descriptions and therefore requiring an equivariant architecture, but the motivation for this is somewhat lacking. Why are these features important? Can you predict binding/ect. properties better with this method? Is there opportunity for guiding generative models?_

Unlike scalar RMSF or isotropic models, anisotropic covariances provide directional information on residue fluctuations, which can be crucial for understanding cryptic pocket formation, hinge motions, or loop breathing—all relevant for binding or allostery. As for downstream utility, we explain a new zero-shot example (section B above) that DynaProt can highlight cryptic pocket formation in an apo structure, suggesting potential for dynamics-aware virtual screening. Thank you for your point about the potential to aid generative models, we hypothesize some use cases above (section B4 above).

- _What precisely is the downstream application of capturing anisotropic covariance matricies? Can you show improvement of these downastream tasks with your method? Eg can you predict various binding affinities or catalytic activates better using your dynamic descriptors? What kind of accuracies are needed to have a significant impact on these downstream tasks. The improvements in table 4 seem minor compared to the reference, do these minor improvements really make an impact? Is the accuracy of the reference already sufficient and the authors are simply accelerating tasks? This significantly effects the significance of the work._

We crafted above sections A, B, and C above to respond to your important points about DynaProt's utility. In the final version of the paper, we would love to incorporate the new tables and figures either in main text or supplemental.

- _Could the authors provide ablations of various aspects of their architecture? How important is the SE(3) equivariance? How dependant are the observed metrics on the length of the MD trajectory that the model is trained on?_

Thank you for the suggestions. We've added new ablations to test both the importance of DynaProt’s Riemannian aware loss (log Frobenius norm) and the SE(3) invariance from the IPA layers. As shown in the new Table 3, removing the log Frobenius norm significantly degrades performance—unsurprising, since we're optimizing over the space of positive definite covariance matrices, which lies on a well-studied Riemannian manifold. Replacing IPA blocks with standard MLPs also hurts performance, suggesting that SE(3) invariance is crucial in our low-data, low-parameter regime. Also with the new experiment on BPTI (Table 2), we see that DynaProt is remarkably able to generalize dynamics to proteins with significantly longer trajectories.

- _How applicable is the method to more dynamic proteins like structurally disordered ones. Can it capture complex dynamic events such as enzymatic breathing motion. What about Allosteric transition? How do your results compare to experimental datasets which show dynamics?_

Due to time constraints, we were unable to retrain DynaProt on datasets that included structurally disordered proteins. But, we did run a new zero-shot ensemble generation on the long-timescale structural dynamics of BPTI, using data from Shaw et al. (2010). BPTI is known to undergo notable conformational rearrangements, including enzymatic breathing motions. As shown in Table 2, DynaProt generalizes remarkably well to this larger and more dynamic system, despite never being trained on it. These results demonstrate DynaProt’s ability to capture rich conformational variability—even beyond the relatively rigid proteins in ATLAS—and provide an exciting step toward modeling phenomena like allostery or disordered regions.

- _What applications do you expect this to work better and worse for?_

We hypothesize DynaProt works best on structured proteins like enzymes where local fluctuations and residue couplings capture meaningful dynamics (see new BPTI experiment). It might struggle on fully disordered regions or massive unfolding events, since it's not trained to model that kind of large-scale changes. We will add this as note in the discussion.

### Reviewer 2 (xnsm)

#### Point by point responses

1. _Clarity / Experiments:_
- _The comparison baseline is relatively limited. While the authors dismiss sequence-only methods as “unfair,” such comparisons (e.g., with ESMDiff or sPROF) would nonetheless contextualize the value of structural input. Even partial evaluations could be informative._

We apologize for not including comparisons to sequence-only methods like ESMDiff and BioEmu in the original submission. We’ve now added these results, and as shown in the updated tables, DynaProt outperforms these methods on nearly all ensemble generation metrics on the Atlas test set (following the evaluation protocol from Jing et al., 2024). We feel that now that structure prediction from sequence is quite reliable with tools like AlphaFold3, the value of structure-based methods like DynaProt is even more clear — one can first predict the structure of their sequence, and then directly input it into DynaProt for fast dynamics predictions.

- _Experiments could be performed on more datasets, including STRUCTURAL DYNAMICS OF BPTI, intrinsically disordered proteins._

Thank you for the suggestion! We have added a new zero-shot conformation generation experiment applying DynaProt to the structural dynamics of BPTI (Shaw et al., 2010), which is known to undergo larger-scale motions than those seen in the ATLAS dataset. Using the evaluation metrics from Jing et al. (2024), we compare the DynaProt-generated ensemble to the full ground truth MD trajectory. As shown in the newly added Table 2, DynaProt performs remarkably well: it achieves high local dynamics accuracy with an RMSF Pearson correlation of 0.88, local anisotropy with RMWD of 0.52, and strong agreement across other metrics such as pairwise RMSD (1.36 Å), W2 distance on PCA projections, and contact-based Jensen-Shannon divergence. These metrics emphasize that DynaProt is able to model both the local flexibility and global ensemble structure with high fidelity. We again thank the reviewer for the suggestion, as this points to DynaProt's ability to generalize to proteins with larger scale dynamics. This experiment will be included as an additional experiment and figure in the main text before the camera ready deadline.


**Table 2**:  **DynaProt** zero-shot ensemble generation on Structural Dynamics of BPTI (Shaw et al (2010)). Evaluation metric protocol from Jing et al (2024).
| Metric                           | **DynaProt (BPTI)** |
|----------------------------------|---------------------|
| Pairwise RMSD (=1.57)            | 1.36                |
| RMSF (=0.84)                     | 0.86                |
| Global RMSF r (↑)                | 0.88                |
| Per-target RMSF r (↑)            | 0.88                |
| Root mean W2-dist Var Contrib (↓)| 0.52                |
| MD PCA W2 (↓)                    | 0.49                |
| Joint PCA W2 (↓)                 | 0.81                |
| Weak contacts J (↑)                 |  0.54               |
Transient contacts J(↑)               | 0.54                |
| # Parameters (↓)                 | 2.86M               |
| Ensemble sampling time (↓)       | ~0.05s              |

More to the reviewer's point (as well as other reviewers) about DynaProt's generalizability, we find that the marginal anisotropic predictions can help reveal cryptic pockets—transient openings on the protein surface that allow ligands to bind (CITE PocketMiner). These cryptic pockets are often the key to making target proteins druggable. We applied DynaProt to zero-shot predict the marginal dynamics of the apo form of ADENYLOSUCCINATE SYNTHETASE (PDB ID: 1ade), a protein featured in the PocketMiner paper’s curated set of cryptic pocket examples. When comparing the predicted anisotropic gaussian blobs overlaid with the holo form (1cib), the predicted Gaussian blobs on the residues involved in the pocket remarkably pointed toward the direction of the holo form protein suggesting that it was indeed capturing the cryptic pocket formation. We really wanted to include this as a figure with the Gaussian blobs overlaid, however NeurIPS 2025’s new rules prevent us from attaching new figures/plots. We plan to include it in the final version or supplementary material, as it’s an exciting use case of DynaProt in action: how dynamic readouts could help uncover druggable motions of enzymes with zero additional supervision.


- _Table 4 lacks visual clarity (e.g., no bolded best-performing numbers). Table 3’s “single rep” column is not clearly explained._

We apologize for the lack of visual clarity in Table 4 — we will bold the best-performing numbers in the camera-ready version of the paper. Unfortunately, formatting edits are not permitted during the rebuttal stage. Additionally, regarding the comment about the “single rep” column in Table 3, we believe this may be a misunderstanding, as there is no such column in that table. If the reviewer is referring instead to Figure 3, we clarify that the “single representation” refers to the per-residue sequence embedding vector  that is updated through the IPA layers — terminology inherited from AlphaFold2. Apologies if we have misunderstood.

2. _Originality / Quality: The method requires a known input structure and does not support conditional or sequence-based generation. While this design enables efficiency, the authors should clarify how it compares or complements recent ensemble-generation models like AlphaFlow._

Thank you for the comment — we will make sure to clarify this in the manuscript. Broadly, methods like AlphaFlow, BioEmu, and ESMDiff are designed for ensemble generation directly from sequence. AlphaFlowMD+Templates is a variant of AlphaFlow that additionally conditions on structure, using it as a template. DynaProt is most comparable to AlphaFlow-MD+Templates, as it also takes a known input structure (assumed as mean conformation). From this structure, DynaProt predicts: (1) per-residue anisotropic marginal covariances (which can be used to visualize residue fluctuations and compute RMSF), (2) pairwise residue couplings, which capture correlated motions across the protein, and (3) ensemble samples by combining (1) and (2) to form a full joint Gaussian over the structure. To this end, we find that due to the recent strides in structure prediction (e.g., AlphaFold3), structure-based models like DynaProt are becoming increasingly necessary. Researchers can now go from sequence → structure with AF3, and then immediately apply DynaProt for fast and detailed dynamics prediction.

We also note that DynaProt is, to our knowledge, the only method that models dynamics through a Gaussian lens (apart from NMA). This not only highlights the novelty of the approach, but also enables extremely efficient sampling and direct readouts of structural dynamics, such as anisotropic residue fluctuations and correlated motions, which are not readily available in existing methods.

3. _Significance: While results outperform NMA, the performance gap with AFMD remains. It’s unclear whether DYNAPROT can fully match distributional realism or functional flexibility prediction._

Insert narrative about DynaProt utility and contribution here

Questions:

1. Could you evaluate DYNAPROT on additional datasets, such as BPTI structural dynamics and IDPs? Could you also include sequence-only baselines (e.g., ESMDiff) for reference?

Thank you for this comment, we have added both zero shot conformation generation experiments on BPTI, and sequence only baselines (ESMDiff, BioEMU) in the new tables 2 and 3 above.

2. Could you provide a full parameter breakdown of your model and baselines? A summary table would help quantify DYNAPROT’s efficiency claims.

TODO

3. Why were AFMD or AlphaFlow methods not evaluated under residue-level flexibility metrics (e.g., RMSF)? Are their ensembles unavailable?

We apologize for this confusion, the AFMD+Templates metrics are listed in Table 4 (as they are readouts from the ensembles). In the final manuscript, we are happy to list it in Table 2 in the main text as well.

4. In Figure 1, what are the axes measuring? (Numerical units, interpretation of values)

Thank you for pointing this out. Figure 1 is intended as a conceptual illustration rather than a quantitative plot to motivate the landscape of current dynamics prediction methods. While DynaProt may output less detailed information than full-blown molecular dynamics simulations, we show that its readouts (marginal/joint Gaussians and ensemble approximations) are still highly useful due to its extreme speed. To clarify: the vertical axis represents the richness or density of dynamics information the method provides, and the horizontal axis represents relative inference speed. 

### Reviewer 3 (J7z4)

#### Point by point responses

Weaknesses:

1. The hierarchical Gaussian parameterization limits the model’s scope and applicability, focusing mainly on local, low-order, and less cooperative motions. See Q1 for general concerns.
2. While the manuscript is clearly presented, it lacks sufficient explanation on the intended use cases where the Gaussian approach would excel. Several important model details are also missing (see Q2).
3. Additional clarifications are needed regarding the benchmark choice and evaluation metrics (see Q3). Some supporting ablation studies are necessary to validate design decisions (see Q4).
4. Current performance seems limited on benchmark datasets and weak on ensemble generation (see Q3).

Questions:
1. Overall concerns:
    
    a. Local dynamics & template dependence: DynaProt focuses on local dynamics, with outputs strongly dependent on the input template, which serves as the mean of its hierarchical Gaussian parameterization. How does template choice affect predictions quality?

    b. Approximation in Gaussian modeling: Using Gaussian distributions may limit expressiveness. How it capture higher-order or multi-residue couplings? Are there approximation errors from heuristic reconstruction (Sec. 3.4)?

    c. Extendability: how would the author extend the model to incorporate side-chain dynamics?

2. Clarification on model details:
    
    a. IPA modification: The original IPA module in AlphaFold2 updates single, pair representation, and SE(3) Frames. What exact modifications were made in DynaProt as shown in Figure 3, and why? For example, why no just use pair representations for residue-residue covariance prediction?

    b. Line 165: It says “these scalar covariances are derived from the full joint covariance matrix via MeanPooling” but in the next section, it seems to directly predicting a scalar for residue i - residue j pair.

    c. How is pairwise covariance decomposition implemented in pytorch? Is it scalable for longer proteins?

3. Experiments and analysis

    a. NMA baseline: How were key parameters (e.g., atom cutoff, gamma) chosen? Will it affects the baseline performance?

    b. It seems the authors trained separate models for marginal (DynaProt-M) and joint covariance (DynaProt-J) prediction? And combined their output for ensemble prediction. Why is this necessary and can one just train a single model for both prediction heads?

    c. Evaluation:
        
        i. ATLAS is known to mainly contains local diversity, which may favor proposed approach. How does DynaProt perform on proteins with large-scale conformation changes (e.g., fast folding proteins)?
        
        ii. Figure 4c: would it be more reasonable to measure residue-residue coupling as the function of spatial distance (or spatial cut-off)? How did AlphaFlow-MD perform on this evaluation?

        iii. Ensemble generation performance: compare Table 4 with Table 1 in the AlphaFlow paper, it seems DynaProt still fall short (vs AFMD without template, sequence only) on most of metrics except for Global RMSF r and Per-taret RMSF r. This is contradicting with their claims that “DynaProt typically outperforms these sequence-only method versions,…”

4. Ablation studies

    a. The authors claimed log-Frobenius distance is better (more stable) than standard Euclidean loss but did not provide empirical evidences.

    b. In Pairwise Dynamics Module, their approach to compute pairwise embedding and update with additional triangular operations seems arbitrary. Has the authors compared with simpler methods such as MLP on the pair repr (if exists) from IPA or on the outer product of single repr?

5. Missing references
    a. Str2Str (https://arxiv.org/abs/2306.03117) is a similar local conformation sampling method using diffusion, using a template structure as input. It should be compared or discussed as a baseline.

    b. Several models missed in discussion on this conformation sampling task: DiG (https://www.nature.com/articles/s42256-024-00837-3) and ConfDiff (https://github.com/bytedance/ConfDiff) are two diffusion based method for protein conformation sampling, the later also provided updated evaluation on ATLAS benchmark. Boltz-2 (https://www.biorxiv.org/content/10.1101/2025.06.14.659707v1) has enhanced conformation diversity trained using conformation data, serving as another plausible baseline.

### Reviewer 4 (Hize)

Quality:

The paper is of average quality. It is well-written, but the experimental section appears lacking. In particular, the evaluations do provide error bars, and the selection of alternative models to benchmark against seems too small. The authors justify this by saying that they intend mainly to benchmark against models that i) take the equilibrium structure as input and ii) explicitly predict covariances.

I don’t see why ii) is justified. It’s usually quite easy to sample many times from a model and compute the empirical covariance. One could argue that takes too much compute for certain models, but this shouldn’t prevent the comparison of estimation accuracy from being carried out at all.

As for i), since DynaProt takes in more information than methods that rely only on sequence, the authors should demonstrate, for at least some of the leading sequence-only models, that DynaProt performs at least as well. I recommend benchmarking against a model like BioEmu, which explicitly tackles equilibrium distributions, samples much faster than AlphaFlow, and is available freely online.

Clarity:

The paper is well-written and generally easy to follow. The one exception is the discussion of the matrix C. I had to read the discussion several times to understand what quantity C is, how it is trained, and how it is interpreted. In particular, since the authors refer to it as a covariance matrix, they should be very explicit about what random variable it is the covariance of. The authors also mention in line 108 that it contains “dynamical couplings” which is “typically” computed as a scalar projection. It’s not clear to me what is meant by dynamical couplings here, and why the word typically is used. Is this procedure followed consistently?

Significance:

I would argue the significance is low to moderate. On one hand, I believe models like this will be superseded very soon by the flurry of research on models that explicitly predict structures. From these models, researchers can trivial estimate covariances. However, the authors do argue that their model is extremely fast. I can imagine this model will have a niche for a couple of years, but as compute gets cheaper, and structure-based equilibrium estimation gets more and more accurate, I do not think this significance will last.

Originality

The model here relies on an AF2 backbone and trains on an open source dataset. I am not personally aware of the specific scheme for composing the 3Nx3N covariance being done before. It is a nice idea overall, but I would say originality is moderate.

Questions:
- How good is the Gaussian approximation? Can you explain more situations where we would need Gaussian covariances but would not need structures? Structural behavior is at least very important to many who study equilibrium distributions, since often one is interested in the biomechanical mechanism underlying, say, drug action.
- Are the structures in Figure 5 connected? It appears that often the beta sheets do not form, and there are chain breaks. Do the authors view this as a problem?
- Do you perform any sequence-similarity based splitting for train/valid/test? Are there any concerns about inadvertent test set poisoning?
- Figure 4A is a nice visualization; would it be possible to repeat this for AlphaFlow/BioEmu/some other structure-based equilibrium predictor?