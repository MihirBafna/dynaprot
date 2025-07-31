### Reviewer 1 (JQtJ)
We appreciate the reviewer’s concerns about the utility of DynaProt's outputs compared against ensemble generation methods and RMSF predictors.They also asked for more insight into the use cases of DynaProt and additional ablations. We have now provided some key experiments (cryptic pocket discovery, dynamics of BPTI, etc.) and explanations to address their concerns.

### **Utility of the predictions**

We start by noting that despite other ensemble generation methods and molecular dynamics toolkits being readily available, NMA has been used ubiquitously for insights into biological discovery (Yamato 2019, Kolossváry 2024, Bahar 2010, Miller 1999). As you mention, enzymatic breathing motion is one of the most prominent examples of a downstream readout commonly extracted from NMA. When structural biologists analyze a protein, they often load its structure into PyMOL and use plugin tools that enable NMA to explore dynamics. 

**Thus, the bar for utility is not to significantly outperform the ensemble generators, but to approach or match them with the speed of normal mode analysis (NMA).**

DynaProt offers a significant upgrade as it produces these local anisotropic predictions faster, with much higher accuracy (main text Tables 2,3,4), and crucially, without requiring simulation or long sampling times (AFMD+Templates). This means that, for any static input structure, DynaProt's marginals gives you not just how much a residue is likely to move, but the direction of its motion—and does so much better than NMA. 

Lastly, even in comparison to ensemble generation methods, DynaProt achieves comparable results with a model that's just <=2.86M parameters and runs in under a second--**a 5 orders of magnitude improvement**. For structural biologists screening thousands of molecules, traditional methods would take 6500*1000 s ~= 75 days. This would be untenable. So there are two main takeaways:

  - DynaProt's utility as a method: _DynaProt's speed and simplicity make  practical and accessible for everyday use in a structural biology toolkit_ (see more explanations below) 
  - DynaProt paper as a perspective for the field: _Perhaps you don’t always need huge models or complex training pipelines to get useful structural dynamics predictions_
  
 In fact, we hope DynaProt encourages the field to branch out from its current reliance on large generative models—sometimes, carefully chosen and faster architectures can be just as effective.

> [Why go ] ... from simple invariant descriptions to include also anisotropic descriptions

Predicting anisotropy per residue encompasses what RMSF provides, but also gives directionality to the residue's fluctuation. This directional information can be immediately useful moreso than RMSF, in cases like cryptic pocket discovery (see below). Cryptic pockets are transient openings on the protein surface that allow ligands to bind (Meller et al 2023, PocketMiner). 

###### DynaProt’s zero-shot cryptic pocket discovery
We applied DynaProt to zero hot predict the marginal dynamics of the apo form of a known synthetase (PDB: 1ADE), from (Meller 2023) curated set of cryptic pockets. When comparing the predicted anisotropic gaussians overlaid with the holo form (1CIB), the predictions on the residues involved in the pocket indeed pointed toward the direction of the holo form protein suggesting that it was indeed capturing the cryptic pocket formation. We really wanted to include this as a figure with the Gaussian blobs overlaid, however NeurIPS 2025’s new rules prevent us from attaching new figures/plots. We'll include it in the final version, highlighting how DynaProt's readouts could help uncover druggable motions of enzymes with zero additional supervision.

> include structural dynamics of BPTI

###### DynaProt modeling the structural dynamics of BPTI (Shaw et al., 2010)
We have added a new zero-shot conformation generation experiment applying DynaProt to the structural dynamics of BPTI, which is known to undergo larger-scale motions than those seen in the ATLAS dataset. Using the evaluation metrics from Jing et al. (2024), we compare the DynaProt-generated ensemble to the full ground truth MD trajectory. As shown in the newly added Table 2, DynaProt performs remarkably well: it achieves high local dynamics accuracy with an RMSF Pearson correlation of 0.88, local anisotropy with RMWD of 0.52, and strong agreement across other metrics such as pairwise RMSD, W2 distance on PCA projections, and contact-based JS divergence. These metrics emphasize that DynaProt is able to model both the local flexibility and global ensemble structure with high fidelity. We again thank the reviewer for the suggestion, as this points to DynaProt's ability to generalize to proteins with larger scale dynamics. This experiment will be included in the revised paper.

**Table 2**:  **DynaProt** zero-shot ensemble generation on dynamics of BPTI.
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



#### Summarized insights about use cases

1. ##### Marginal covariances can be used as fast replacement to current NMA plugins
    As discussed above, we believe DynaProt’s ability to predict marginal covariances with high accuracy efficiency makes it a practical replacement for current Normal Mode Analysis (NMA) plugins commonly used in PyMOL, especially for use cases like cryptic pocket discovery. 

2. ##### DynaProt’s pair residue coupling and ensemble generation
    DynaProt also predicts pairwise residue couplings that indicate how each pair of residues move in a correlated fashion and can generate reasonable structure ensembles much faster than any current method. Despite being significantly smaller in parameter count, DynaProt generates ensembles with higher fidelity than many sequence-based methods (see Table 1) and structure/template-based methods (see Table 2), and is competitive with SOTA approaches like AFMD+Templates. We hypothesize that this could change how researchers interact with structures—imagine loading a structure in PyMOL and instantly getting a plausible ensemble.

3. ##### DynaProt's outputs as future use for design
    Finally, we believe DynaProt's readouts could enable dynamics-conditioned design. For instance, users could specify the anisotropic marginals of regions that should remain rigid or flexible (and with directionality)—e.g., stabilizing a loop or introducing breathing motion near a pocket. Models like ProtComposer (Stark and Jing et al 2025), which condition on geometric ellipsoids to guide secondary structure, could instead condition on DynaProt’s Gaussian ellipsoids to design proteins that exhibit desired dynamic behaviors. The pairwise coupling readouts could also be used to design proteins with coupled movements. We plan to add these ideas in the discussion as future work.


#### Ablations 
> Q: Could the authors provide ablations of various aspects of their architecture? How important is the SE(3) equivariance? How dependant are the observed metrics on the length of the MD trajectory that the model is trained on?

Thank you for the suggestions. We've added new ablations to test both the importance of DynaProt’s Riemannian aware loss (log Frobenius norm) and the SE(3) invariance from the IPA layers. As shown in the new Table 3, removing the log Frobenius norm significantly degrades performance—unsurprising, since we're optimizing over the space of positive definite covariance matrices, which lies on a well-studied Riemannian manifold. Replacing IPA blocks with standard MLPs also hurts performance, suggesting that SE(3) invariance is crucial in our low-data, low-parameter regime. Also with the new experiment on BPTI (Table 2), we see that DynaProt is remarkably able to generalize dynamics to proteins with significantly longer trajectories.
    
**Table 3**: **DynaProt** ablations. Replacing Riemannian aware LogFrob loss with canonical MSE, and replacing IPA layers (SE(3) invariance) with MLP blocks. Comparisons on the marginal Gaussian predictions and residue level RMSF Pearson correlation.
| Metric           | DynaProt-M | No LogFrob Loss | No SE(3) Invariance |
|------------------|----------|------------------|----------------------|
| RMWD Var (↓)     | **1.18**     | 2.70             | 1.92                 |
| Sym KL Var (↓)   | **0.91**     | 9.26             | 4.46                |
| RMSF r (↑)       | **0.87**    | 0.38             | 0.48                 |


#### Other Questions
> Q: The authors do little to investigate how well their method predicts large scale protein restructuring ... What applications do you expect this to work better and worse for?

We hypothesize DynaProt works best on structured proteins like enzymes where local fluctuations and residue couplings capture meaningful dynamics (see new zero shot BPTI experiment). It might struggle on fully disordered regions or massive unfolding events, since it's not trained to model that kind of large-scale changes. We will add this as note in the discussion.


---
### Reviewer 2 (xnsm)

#### Additional baseline methods

We’ve now added additional basline experiment results for both sequence based methods (Table 1) and an additional structure/template based method (ConDiff in Table 4), and as shown in the updated tables, DynaProt matches/outperforms these methods on nearly all ensemble generation metrics on the ATLAS test set (following the evaluation protocol from Jing et al., 2024) and is comparable to AFMD. Unfortunately, when attempting to generate ensembles on the ATLAS test set using Str2Str, another structure-based ensemble generation method, we encountered out of memory errors — even on NVIDIA A100 80GB GPUs with a batch size 1. This suggests that ATLAS proteins may be too large for Str2Str to handle in its current form, making a direct comparison infeasible at this time.

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


**Table 4**: Comparison of **DynaProt** ensemble generation (on ATLAS test set from Jing et al (2024)) against additional structure/template based method ConfDiff.

| Metric                         | **DynaProt**  | ConfDiff-ESM-r3-MD |
|-------------------------------|--------------|----------|
| Pairwise RMSD (=2.89)         | **2.17**         | 3.91     |
| RMSF (=0.1.48)                | **1.10**         | 2.79     | 
| Global RMSF r (↑)             | **0.71**     | 0.48     |     
| Per-target RMSF r (↑)         | **0.86**     | 0.82     |    
| Root mean W2-dist Var Contrib (↓) | **1.18**  | 1.51     |   
| MD PCA W2 (↓)                 | 1.74  | **1.66**     |    
| Joint PCA W2 (↓)              |**2.39**  | 2.89     |   
| Weak contacts J (↑)           | 0.51         | **0.56**     |
| Transient contacts J (↑)      | 0.29  |**0.34**     | 
| # Parameters (↓)              | **2.86M**    | -     |   
| Ensemble sampling time (↓)    | **~0.14s**   | ~570s    | 

#### Experiments on BPTI and other datasets

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

#### Relation to AlphaFlow

Bowen's response:
>The emphasis of our work is to, given an input structure, provide a fast but accurate way of previewing its structural dynamics. AlphaFlow and other ensemble generators aim to provide a comprehensive view. The latter approach is higher-fidelity but much slower in practice. For the same reason that NMA analysis continued to be useful despite classical ensemble generation approaches like MD, we think DynaProt will complement AlphaFlow...

Mihir's Response:
>Thank you for the comment — we will make sure to clarify this in the manuscript. Broadly, methods like AlphaFlow, BioEmu, and ESMDiff are designed for ensemble generation directly from sequence. AlphaFlowMD+Templates is a variant of AlphaFlow that additionally conditions on structure, using it as a template. DynaProt is most comparable to AlphaFlow-MD+Templates, as it also takes a known input structure (assumed as mean conformation). From this structure, DynaProt predicts: (1) per-residue anisotropic marginal covariances (which can be used to visualize residue fluctuations and compute RMSF), (2) pairwise residue couplings, which capture correlated motions across the protein, and (3) ensemble samples by combining (1) and (2) to form a full joint Gaussian over the structure. To this end, we find that due to the recent strides in structure prediction (e.g., AlphaFold3), structure-based models like DynaProt are becoming increasingly necessary. Researchers can now go from sequence → structure with AF3, and then immediately apply DynaProt for fast and detailed dynamics prediction.

>We also note that DynaProt is, to our knowledge, the only method that models dynamics through a Gaussian lens (apart from NMA). This not only highlights the novelty of the approach, but also enables extremely efficient sampling and direct readouts of structural dynamics, such as anisotropic residue fluctuations and correlated motions, which are not readily available in existing methods.



#### Other questions

> Table 4 lacks visual clarity (e.g., no bolded best-performing numbers). Table 3’s “single rep” column is not clearly explained.

We apologize for the lack of visual clarity in Table 4--we will bold the best-performing numbers in the camera-ready version of the paper. Unfortunately, formatting edits are not permitted during the rebuttal stage. Additionally, regarding the comment about the “single rep” column in Table 3, we believe this may be a misunderstanding, as there is no such column in that table. If the reviewer is referring instead to Figure 3, we clarify that the “single representation” refers to the per-residue sequence embedding vector  that is updated through the IPA layers--terminology inherited from AlphaFold2. Apologies if we have misunderstood.

> Why were AFMD or AlphaFlow methods not evaluated under residue-level flexibility metrics (e.g., RMSF)? 

We apologize for this confusion, the AFMD+Templates metrics are listed in Table 4 (as they are readouts from the ensembles). In the final manuscript, we are happy to list it in Table 2 in the main text as well.

> In Figure 1, what are the axes measuring?

Thank you for pointing this out. Figure 1 is intended as a conceptual illustration rather than a quantitative plot to motivate the landscape of current dynamics prediction methods. While DynaProt may output less detailed information than full-blown molecular dynamics simulations, we show that its readouts (marginal/joint Gaussians and ensemble approximations) are still highly useful due to its extreme speed. To clarify: the vertical axis represents the richness or density of dynamics information the method provides, and the horizontal axis represents how fast each method is relative to another. 

---
### Reviewer 3 (J7z4)

#### Q1: Scope and applicability of the Gaussian parameterization

> Outputs strongly dependent on the input template...How does template choice affect predictions quality?

Thank you for the question. To improve generalizability and reduce template bias, we actually train DynaProt with data augmentations—specifically, by randomly sampling input frames from the ATLAS trajectory during training. We apologize for not making this clearer in the main text and will clarify in the revision. At inference time, we use the PDB entry as the "mean" structure, and DynaProt generalizes well despite this, as seen in our current results.


> How it capture higher-order or multi-residue couplings? Are there approximation errors from heuristic reconstruction?

Great question. You're right that higher-order couplings are tough to model directly. That said, we approximate them through the set of all pairwise couplings, which DynaProt predicts explicitly. Especially for local interactions, this works quite well—as shown in Figure 4C, DynaProt outperforms NMA. And in our new experiments (see tables below), DynaProt also captures transient contacts better than BioEmu. TODO: approximation errors

> how would the author extend the model to incorporate side-chain dynamics?

TODO:

#### Q2: Model details

> IPA modification ... and why not use pair rep for residue-residue coupling?
> It seems the authors trained separate models for marginal (DynaProt-M) and joint covariance (DynaProt-J) prediction? ... Why is this necessary and can one just train a single model for both prediction heads?

The only modification we made was disabling the pair representation updates in IPA. For marginal predictions (DynaProt-M), we found pair features unnecessary and wanted to keep the model lightweight. But you're right--pair representations are useful for predicting residue-residue couplings, which is why we include them in DynaProt-J pairformer style model. This lets us disentangle the two tasks and avoid extra computation when it's not needed.

> Line 165: It says “these scalar covariances are derived from the full joint covariance matrix via MeanPooling” but in the next section, it seems to directly predicting a scalar for residue i - residue j pair.

We apologize for the confusion, during the initial preprocessing of the ATLAS MD data ( when creating training labels), we compute the i–j coupling by taking the average of the values in the corresponding 3×3 block of the full joint covariance matrix. At prediction time, DynaProt directly outputs a scalar coupling value for each residue i–j pair.

> How is pairwise covariance decomposition implemented in pytorch? Is it scalable for longer proteins?

Apologies for our misunderstanding, but we want to clarify what is meant by "pairwise covariance decomposition"? Is this referring to the heuristic (eq 5) where we compose the marginal cholesky factors with the pairwise coupling yielding the full joint? Regardless, we do find it all implementations of DynaProt scalable to large proteins as we are able to efficiently preprocess and real time compute every protein in ATLAS (which has sometimes >1000 residues).



#### Q3: Experiments (and additional baselines)

> NMA baseline: How were key parameters (e.g., atom cutoff, gamma) chosen?

We used the same NMA parameters as those in Jing et al. (2024) for our Anisotropic Network Model instantation for NMA. This ensures a faithful and fair comparison, as their setup has been validated in essentially the exact same dynamics benchmarking tasks.

> ATLAS is known to mainly contains local diversity, which may favor proposed approach. How does DynaProt perform on proteins with large-scale conformation changes (e.g., fast folding proteins)?

Though we did not have time to evaluate DynaProt on all fast-folding proteins, we have added a new zero-shot conformation generation experiment applying DynaProt to the structural dynamics of BPTI, which is known to undergo larger-scale motions than those seen in the ATLAS dataset. Using the evaluation metrics from Jing et al. (2024), we compare the DynaProt-generated ensemble to the full ground truth MD trajectory. As shown in the newly added Table 2, DynaProt performs remarkably well: it achieves high local dynamics accuracy with an RMSF Pearson correlation of 0.88, local anisotropy with RMWD of 0.52, and strong agreement across other metrics such as pairwise RMSD, W2 distance on PCA projections, and contact-based JS divergence. These metrics emphasize that DynaProt is able to model both the local flexibility and global ensemble structure with high fidelity. We again thank the reviewer for the suggestion, as this points to DynaProt's ability to generalize to proteins with larger scale dynamics. This experiment will be included in the revised paper.

**Table 2**:  **DynaProt** zero-shot ensemble generation on dynamics of BPTI.
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

More to the reviewer's point (as well as other reviewers) about DynaProt's generalizability, we find that the marginal anisotropic predictions can help reveal cryptic pockets—transient openings on the protein surface that allow ligands to bind (CITE PocketMiner). These cryptic pockets are often the key to making target proteins druggable. We applied DynaProt to zero-shot predict the marginal dynamics of the apo form of ADENYLOSUCCINATE SYNTHETASE (PDB ID: 1ADE), a protein featured in the PocketMiner paper’s curated set of cryptic pocket examples. When comparing the predicted anisotropic gaussian blobs overlaid with the holo form (1CIB), the predicted Gaussian blobs on the residues involved in the pocket remarkably pointed toward the direction of the holo form protein suggesting that it was indeed capturing the cryptic pocket formation. We really wanted to include this as a figure with the Gaussian blobs overlaid, however NeurIPS 2025’s new rules prevent us from attaching new figures/plots. We plan to include it in the final version or supplementary material, as it’s an exciting use case of DynaProt in action: how dynamic readouts could help uncover druggable motions of enzymes with zero additional supervision.

> Contradicting with their claims that “DynaProt typically outperforms these sequence-only method versions,…”

Thank you, the reviewer brings up a good point about the evaluations to sequence only methods, we here provide more evaluations on sequence only methods (as also requested by other reviewers) DynaProt matches/outperforms these methods on nearly all ensemble generation metrics on the ATLAS test set (following the evaluation protocol from Jing et al., 2024) and is comparable to AFMD. We will change the wording in text to reflect this as well.

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


#### Q4: ablations

We've added new ablations to test both the importance of DynaProt’s Riemannian aware loss (log Frobenius norm) and the SE(3) invariance from the IPA layers. As shown in the new Table 3, removing the log Frobenius norm significantly degrades performance—unsurprising, since we're optimizing over the space of positive definite covariance matrices, which lies on a well-studied Riemannian manifold. Replacing IPA blocks with standard MLPs also hurts performance, suggesting that SE(3) invariance is crucial in our low-data, low-parameter regime. Also with the new experiment on BPTI (Table 2), we see that DynaProt is remarkably able to generalize dynamics to proteins with significantly longer trajectories.
    
**Table 3**: **DynaProt** ablations. Replacing Riemannian aware LogFrob loss with canonical MSE, and replacing IPA layers (SE(3) invariance) with MLP blocks. Comparisons on the marginal Gaussian predictions and residue level RMSF Pearson correlation.
| Metric           | DynaProt-M | No LogFrob Loss | No SE(3) Invariance |
|------------------|----------|------------------|----------------------|
| RMWD Var (↓)     | **1.18**     | 2.70             | 1.92                 |
| Sym KL Var (↓)   | **0.91**     | 9.26             | 4.46                |
| RMSF r (↑)       | **0.87**    | 0.38             | 0.48                 |

#### Q5: missing references


---
### Reviewer 4 (Hize)


#### Improvements of experimental section

 - ##### Clarification about Reviewer's point (ii)
    > I don’t see why ii) is justified. It’s usually quite easy to sample many times from a model and compute the empirical covariance. One could argue that takes too much compute for certain models, but this shouldn’t prevent the comparison of estimation accuracy from being carried out at all.

    We apologize for the confusion, actually we do not limit our comparisons against methods that "explicitly predict covariances". AFMD+Templates generates ensembles from which, as you mention, we can derive the covariances and we directly compare against these in the main text (Tables 3 and 4). That said, your point about needing additional experiments/baselines is well taken -- we have added them below.


- ##### Comparisons against additional sequence-input methods and structure/template based methods

    Thank you to the reviewer (and other reviewers) regarding this point about additional baseline comparisons. We’ve now added results for both sequence based methods (Table 1) and an additional structure/template based method (ConDiff in Table 4), and as shown in the updated tables, DynaProt matches/outperforms these methods on nearly all ensemble generation metrics on the ATLAS test set (following the evaluation protocol from Jing et al., 2024) and is comparable to AFMD. Unfortunately, when attempting to generate ensembles on the ATLAS test set using Str2Str, another structure-based ensemble generation method, we encountered out of memory errors — even on NVIDIA A100 80GB GPUs with a batch size 1. This suggests that ATLAS proteins may be too large for Str2Str to handle in its current form, making a direct comparison infeasible at this time.

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


**Table 4**: Comparison of **DynaProt** ensemble generation (on ATLAS test set from Jing et al (2024)) against additional structure/template based method ConfDiff.

| Metric                         | **DynaProt**  | ConfDiff-ESM-r3-MD |
|-------------------------------|--------------|----------|
| Pairwise RMSD (=2.89)         | **2.17**         | 3.91     |
| RMSF (=0.1.48)                | **1.10**         | 2.79     | 
| Global RMSF r (↑)             | **0.71**     | 0.48     |     
| Per-target RMSF r (↑)         | **0.86**     | 0.82     |    
| Root mean W2-dist Var Contrib (↓) | **1.18**  | 1.51     |   
| MD PCA W2 (↓)                 | 1.74  | **1.66**     |    
| Joint PCA W2 (↓)              |**2.39**  | 2.89     |   
| Weak contacts J (↑)           | 0.51         | **0.56**     |
| Transient contacts J (↑)      | 0.29  |**0.34**     | 
| # Parameters (↓)              | **2.86M**    | -     |   
| Ensemble sampling time (↓)    | **~0.14s**   | ~570s    | 

- ##### Additional experiments on BPTI and cryptic pocket discovery


#### Clarification of the matrix C.

Apologies for the confusion about the residue coupling matrix \( C \). Simply put, each entry \( C_{ij} \) tells us how the movement of residue *i* relates to the movement of residue *j*. It measures how much their positions tend to move together. To compute this from the existing MD ensembles, we start from the full joint covariance matrix (which is size \( 3N \times 3N \)) that captures how all residue positions co-vary in 3D space. Each \( 3 \times 3 \) block in that matrix shows how the XYZ coordinates of residue *i* covary with residue *j*. We then reduce each of these blocks to a single number -- usually by taking the average of the values -- to get the final \( N \times N \) matrix \( C \).That’s what we meant by "typically"--averaging is the default, but other ways of summarizing the 3x3 block are possible. We will clarify this in the revised version of the paper.

#### Significance of fast dynamics predictors
> Reviewer: I would argue the significance is low to moderate. 

We start by noting that despite other ensemble generation methods and molecular dynamics toolkits being readily available, NMA has been used ubiquitously for insights into biological discovery (CITE MANY PAPERS HERE), When structural biologists analyze a protein, they often load its structure into PyMOL and use plugin tools that enable NMA to explore dynamics. 

**Thus, the bar for utility is not to significantly outperform the ensemble generators, but to approach or match them with the speed of normal mode analysis (NMA).**

DynaProt offers a significant upgrade as it produces these local anisotropic predictions faster, with much higher accuracy (main text Tables 2,3,4), and crucially, without requiring simulation or long sampling times (AFMD+Templates). This means that, for any static input structure, DynaProt's marginals gives you not just how much a residue is likely to move, but the direction of its motion—and does so much better than NMA.

Lastly, even in comparison to ensemble generation emthods, DynaProt achieves comparable results with a model that's just <=2.86M parameters and runs in under a second--**a 5 orders of magnitude improvement**. For structural biologists screening thousands of molecules, traditional methods would take 6500*1000 s ~= 75 days. This would be untenable. So there are two main takeaways:

  - DynaProt's utility as a method: _DynaProt's speed and simplicity make  practical and accessible for everyday use in a structural biology toolkit_ (see more explanations below) 
  - DynaProt paper as a perspective for the field: _Perhaps you don’t always need huge models or complex training pipelines to get useful structural dynamics predictions_
  
  In fact, we hope DynaProt encourages the field to branch out from its current reliance on large generative models—sometimes, carefully chosen and faster architectures can be just as effective.



#### Other questions

> How good is the Gaussian approximation? Can you explain more situations where we would need Gaussian covariances but would not need structures? 

We apologize if we have misunderstood the question, but we do indeed use the structure (the covariance matrices are not studied in isolation). After DynaProt predicts the marginal covariances, they can be visualized on top of the existing/input equilibrium structure easily in PyMol for further analysis as we have shown in Figure 4A. We have further use cases of these marginal gaussian predictions for cryptic pocket discovery explained above.

> Are the structures in Figure 5 connected? It appears that often the beta sheets do not form, and there are chain breaks.

We attribute this to the fact that DynaProt outputs only Cα coordinates, so tools like PULCHRA are needed to reconstruct full backbones, which occasionally are imperfect resulting in improper secondary structure classification.

>Do you perform any sequence-similarity based splitting for train/valid/test?

We use the ATLAS train/val/test split exactly as described in Jing et al 2024, to fairly compare our method against others, as all other methods (trained on ATLAS) have used exactly the same split. To our knowledge, Jing et al 2024 did not use any sequence-similarity based cutoffs, opting for PDB deposit date cutoffs instead.

>Figure 4A is a nice visualization; would it be possible to repeat this for AlphaFlow/BioEmu/some other structure-based equilibrium predictor?

We would love to include these additional figures for AlphaFlow predicted marginals, however this year's NeurIPS format does not permit us to supply figures, links, or pdfs. These plots can definitely be added to the final camera ready version if accepted.
