# Lagrangian-Discovery-Stochastic-Systems
Data-driven discovery of interpretable Lagrangian of stochastically excited dynamical systems


# Lagrange_Discovery_CPC
Discovering interpretable Lagrangian of dynamical systems from data

This repository contains the python codes of the paper 
  > + Tapas Tripura and Souvik Chakraborty (2023). Discovering interpretable Lagrangian of dynamical systems from data. [Paper](https://doi.org/10.1016/j.cpc.2023.108960)

![Lagrangian Discovery](framework.jpg)

# Files
  + `Discovery_Harmonic.py` Python file for discovering Lagrangian of stochastic Harmonic oscilator.
  + `Discovery_Pendulum.py` Python file for discovering Lagrangian of stochastic Pendulum.
  + `Discovery_Pendulum_missing.py` Python file for discovering Lagrangian of stochastic Pendulum with missing harmonic basses in library.
  + `Discovery_Duffing.py` Python file for discovering Lagrangian of stochastic Duffing oscillator.
  + `Discovery_3DOF.py` Python file for discovering Lagrangian of stochastic 3DOF oscillator.
  + `Discovery_string.py` Python file for discovering Lagrangian of stochastic Wave equation.
  + `Discovery_Blade.py` Python file for discovering Lagrangian of stochastic Euler-Bernoulli Beam.
  + `utils.py` Python file containing useful functions for sparse regression and Euler-Lagrange library construction.
  + `utils_data.py` Python file containing useful functions for solving SDEs/SPDEs.
  + `Figure_basis.py` Python file for generating stem plots of basis functions.
  + `Figure_hamiltonian.py` Python file for generating plots of Hamiltonian.
  + `Figure_response.py` Python file for generating plots of system responses.
  + `beam3fun.py` Python file containing functions of Newmark-beta and assembling Beam elements.

# BibTex
Please cite us at,
```
@article{tripura2024data,
  title={Data-driven discovery of interpretable Lagrangian of stochastically excited dynamical systems},
  author={Tripura, Tapas and Panda, Satyam and Hazra, Budhaditya and Chakraborty, Souvik},
  journal={arXiv preprint arXiv:2402.17122},
  year={2024}
}
```
