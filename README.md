# Benchmarking Black Box Optimization Algorithms
A collection of black-box optimization algorithms from various sources. Utilized in the paper "Benchmarking Derivative-Free Global Optimization Algorithms under Limited Dimensions and Large Evaluation Budgets". Beware that some of the algorithms require special installations and/or licenses.

- **Script_One_Function.m** -  describes the basic call for the algorithms on a single problem (**objective_function.m**). All the algorithms were modified to have the same input-output structure.
- **Script_To_Run_All_Experiments.m** - script that can be used to generate the results described in the paper. This script also downloads the instance files from the DIRECTGOLib: https://github.com/blockchain-group/DIRECTGOLib
- **Script_Data_To_IOHanalyzer.m** - script that transforms the data logs into files that can be analyzed in the IOHanalyzer (https://iohanalyzer.liacs.nl/). This script also downloads the data files that contain the experimental results presented in the paper: https://doi.org/10.5281/zenodo.8362954
