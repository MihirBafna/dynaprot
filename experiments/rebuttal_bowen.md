### Reviewer 1 (JQtJ)

#### Utility of the predictions 

#### Motivation for anisotropic descriptors

#### Further insights about use cases

#### Ablations 


### Reviewer 2 (xnsm)

#### Additional baseline methods


#### Experiments on BPTI


#### Relation to AlphaFlow

#### Other questions

> Table 3’s “single rep” column is not clearly explained.

> Why were AFMD or AlphaFlow methods not evaluated under residue-level flexibility metrics (e.g., RMSF)? 

> In Figure 1, what are the axes measuring?

### Reviewer 3 (J7z4)

#### Q1: Scope and applicability of the Gaussian parameterization

> outputs strongly dependent on the input template...How does template choice affect predictions quality?

> How it capture higher-order or multi-residue couplings? Are there approximation errors from heuristic reconstruction?

> how would the author extend the model to incorporate side-chain dynamics?

#### Q2: model details


#### Q3: Further baselines

#### Q4: ablations


#### Q5: missing references



### Reviewer 4 (Hize)


### Comparisons against sequence-input methods

[BioEmu results from Boltz...]

> ii) explicitly predict covariances.

### Clarification of the matrix C.

### Significance of fast dynamics predictors

> How good is the Gaussian approximation? Can you explain more situations where we would need Gaussian covariances but would not need structures? 

### Other questions

>Are the structures in Figure 5 connected? It appears that often the beta sheets do not form, and there are chain breaks.

>Do you perform any sequence-similarity based splitting for train/valid/test?

>Figure 4A is a nice visualization; would it be possible to repeat this for AlphaFlow/BioEmu/some other structure-based equilibrium predictor?
