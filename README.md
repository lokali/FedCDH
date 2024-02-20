# [ICLR 2024] Federated Causal Discovery from Heterogeneous Data

This is the official implementation of the paper **"[Federated Causal Discovery from Heterogeneous Data](https://openreview.net/forum?id=m7tJxajC3G)"**, ICLR 2024.


If you find it useful, please consider citing:
```
@inproceedings{li2024learning,
  title={Learning to Optimize Permutation Flow Shop Scheduling via Graph-based Imitation Learning},
  author={Li, Loka and Ng, Ignavier and Luo, Gongxu and Huang, Biwei and Chen, Guangyi and Liu, Tongliang and Gu, Bin and Zhang, Kun},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```


## Overview

- In this paper, we propose FedCDH, a novel constraint-based approach for federated causal discovery from heterogeneous data. The figure below exhibits the overall framework of our FedCDH. 

![FedCDH](FedCDH.jpg)




## How to Run

- Installation: R package.
  - Our federated conditional independet test method is developed based on [R Package](https://github.com/ericstrobl/RCIT), please follow their procedures and install all the dependencies at first. 

- Installation: Environment.
```sh
# Set up a new conda environment with Python 3.8.
conda create -n FedCDH python=3.8
conda activate FedCDH

# Please navigate to the root directory, and install our source code.
pip install -e . 

# Install other python libraries.
pip install causaldag rpy2 numpy scipy tqdm networkx 
```

- Evaluation: quick start.
```sh
# Parameters:
#     N: number of instances to evaluate
#     d: number of variables
#     K: number of clients
#     n: number of samples in one client
#     model: data generation model, linear or general.
cd tests
python TestFedCDH.py --N 10 --d 6 --K 10 --n 100 --model linear  
```



## Acknowledgements


We would like to sincerely thank these related works and open-sourced codes which inspired us:

- Causal-learn package: [https://github.com/py-why/causal-learn](https://github.com/py-why/causal-learn).
- CausalDAG package: [https://uhlerlab.github.io/causaldag](https://uhlerlab.github.io/causaldag/).
- Huang et al., Causal Discovery from Heterogeneous/Nonstationary Data with Independent Changes. JMLR, 2020.
  - Code:  [https://github.com/Biwei-Huang/Causal-Discovery-from-Nonstationary-Heterogeneous-Data](https://github.com/Biwei-Huang/Causal-Discovery-from-Nonstationary-Heterogeneous-Data).
- Zhang et al., Kernel-based Conditional Independence Test and Application in Causal Discovery. UAI, 2011.
  - Code:  [https://causal-learn.readthedocs.io/en/latest/independence_tests_index/kci.html](https://causal-learn.readthedocs.io/en/latest/independence_tests_index/kci.html). 
-  Strobl et al., Approximate Kernel-Based Conditional Independence Tests for Fast Non-Parametric Causal Discovery. Journal of Causal Inference, 2018.
    - Code: [https://github.com/ericstrobl/RCIT](https://github.com/ericstrobl/RCIT).

And we also sincerely thank the authors of the baseline methods for making their source codes public:

- Gao et al., FedDAG: Federated DAG Structure Learning. TMLR, 2022.
  - Code: [https://github.com/ErdunGAO/FedDAG](https://github.com/ErdunGAO/FedDAG).
- Ng et al., Towards Federated Bayesian Network Structure Learning with Continuous Optimization. AISTATS, 2022.
  - Code: [https://github.com/ignavierng/notears-admm](https://github.com/ignavierng/notears-admm).

