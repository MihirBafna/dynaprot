### Reviewer 1 (JQtJ)

#### Utility of the predictions

1. WRT to ensemble generators

2. WRT RMSF predictors



The utility of our method is precisely that of normal mode analysis... 

list papers using NMA for biological discovery.

As you mention, enzymatic breathing motion is one of the most prominent examples of a downstream readout commonly extracted from NMA.

The bar for utility is **not** to significantly outperform the ensemble generators, but to approach or match them with the speed of normal mode analysis.

Six orders of mangnitude improvement

To our knowledge, we are the first work to attempt this kind of framing - normal modes are far more useful than scalar RMSF predictions.

As an example we provide a cryptic pocket and BPTI case study below.

#### Insights about use cases



#### Ablations 


### Reviewer 2 (xnsm)

#### Additional baseline methods

Bash the sequence only methods

#### Experiments on BPTI

#### Relation to AlphaFlow

The emphasis of our work is to, given an input structure, provide a fast but accurate way of previewing its structural dynamics. AlphaFlow and other ensemble generators aim to provide a comprehensive view. The latter approach is higher-fidelity but much slower in practice. For the same reason that NMA analysis continued to be useful despite classical ensemble generation approaches like MD, we think DynaProt will complement AlphaFlow...


#### Other questions

> Table 3’s “single rep” column is not clearly explained.

> Why were AFMD or AlphaFlow methods not evaluated under residue-level flexibility metrics (e.g., RMSF)? 

> In Figure 1, what are the axes measuring?

### Reviewer 3 (J7z4)

#### Q1: Scope and applicability of the Gaussian parameterization

> outputs strongly dependent on the input template...How does template choice affect predictions quality?

> How it capture higher-order or multi-residue couplings? Are there approximation errors from heuristic reconstruction?

Our coupling is better than NMA.

We get transient contacts better than BioEmu. This solely arises from the prediction of couplings (since marginal Gaussians)

> how would the author extend the model to incorporate side-chain dynamics?

#### Q2: model details


#### Q3: Further baselines

>“DynaProt typically outperforms these sequence-only method versions,…”




#### Q4: ablations


#### Q5: missing references



### Reviewer 4 (Hize)


### Comparisons against sequence-input methods

> ii) explicitly predict covariances.

### Clarification of the matrix C.

### Significance of fast dynamics predictors

> How good is the Gaussian approximation? Can you explain more situations where we would need Gaussian covariances but would not need structures? 

### Other questions

>Are the structures in Figure 5 connected? It appears that often the beta sheets do not form, and there are chain breaks.

>Do you perform any sequence-similarity based splitting for train/valid/test?

>Figure 4A is a nice visualization; would it be possible to repeat this for AlphaFlow/BioEmu/some other structure-based equilibrium predictor?
