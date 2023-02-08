# Matrix-based-Dependence
code of "Measuring the Dependence with Matrix-based Entropy Functional" in AAAI 2021

We provide Pytorch implementation to design to robust loss function for learning under covariate shift and noisy environments (the SECOND application); by minimizing the mutual information (NOT normalized mutual information) between the distribution of prediction residual p(e) and input distribution p(x).

We also provide MATLAB code to implement (normalized) total correlation and (normalized) dual total correlation.

Note that,  the use of normalization depends on the priority given to interpretability. 
For example, when the measure is employed as a loss function, the normalization does not contribute to performance. 
However, if we want an interpretable dependence measure lies between 0 and 1, normalization is required.
