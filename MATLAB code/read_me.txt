"demo_NTC_estimation" provides demo to evaluate "total correlation" (Eq. 2) and "normalized total correlation" (Eq. 7) in the paper, 
given 5 independent Gaussian variables with ground-truth dependence value 0.

"demo_NDTC_estimation" provides demo to evaluate "dual total correlation" (Eq. 5) and "normalized dual total correlation" (Eq. 8) in the paper, 
also given 5 independent Gaussian variables with ground-truth dependence value 0.

Note that,  the use of normalization depends on the priority given to interpretability. 
For example, when the measure is employed as a loss function, the normalization does not contribute to performance. 
However, if we want an interpretable dependence measure lies between 0 and 1, normalization is required.