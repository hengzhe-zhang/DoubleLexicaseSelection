# A Double Lexicase Selection Operator for Bloat Control in Evolutionary Feature Construction for Regression

This repository contains a minimal implementation of our method which uses a double-stage lexicase selection operator to control bloat in evolutionary feature construction methods for regression tasks. This operator consists of a two-stage selection process, selecting individuals based on fitness values in the first stage and based on tree sizes in the second stage.

## Usage 
- 📌 Clone the repository and run DLS_GP.py to utilize the code. 
- 📊 The code uses the diabetes dataset for training and testing, which is standardized before use. 
- 🧬 Genetic Programming is used to evolve a population of symbolic models, evaluated based on mean squared error on the training data. 
- 🔄 The provided selection function `doubleLexicase` controls the bloat. 
- 🏆 The performance of the proposed selection operator is compared with the traditional lexicase selection operator.
## Lightspots
- 🚀 The double lexicase selection operator effectively controls bloat in symbolic regression / evolutionary feature construction for regression tasks.
- 💪 Compared to the traditional method of setting a depth limit, our approach significantly reduces the sizes of constructed features across all datasets, while maintaining a similar level of predictive performance.
- 🌟 The double lexicase selection operator achieves a good trade-off between model performance and model size.
## Dependencies

The code is written in Python and requires the following dependencies:
- 🐍 DEAP
- 🔢 NumPy
- 🧬 scikit-learn
## Reference

This code is based on the GECCO 2023 paper "A Double Lexicase Selection Operator for Bloat Control in Evolutionary Feature Construction for Regression". Please cite our paper if you find it helpful!

```bibtex
@inproceedings{zhang2023double,
  title={A Double Lexicase Selection Operator for Bloat Control in Evolutionary Feature Construction for Regression},
  author={Zhang, Hengzhe and Chen, Qi and Xue, Bing and Banzhaf, Wolfgang and Zhang, Mengjie},
  booktitle={Proceedings of the Genetic and Evolutionary Computation Conference},
  pages={1194--1202},
  year={2023}
}
```
