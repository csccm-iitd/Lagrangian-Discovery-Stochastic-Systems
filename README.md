# Lagrangian-Discovery-Stochastic-Systems
Data-driven discovery of interpretable Lagrangian of stochastically excited dynamical systems
```
This paper delves into Lagrangian discovery for conservative and non-conservative systems under stochastic excitation. The study critically evaluates recent computational methodologies for learning Lagrangians from observed data, highlighting the limitations in interpretability and the exclusion of stochastic excitation. To address these gaps, an automated data-driven framework is proposed for the simultaneous discovery of Lagrange densities and diffusion coefficients of stochastically excited dynamical systems by leveraging sparse regression. This novel framework offers several advantages over existing approaches.
  + Firstly, it provides an interpretable description of the underlying Lagrange density, allowing for a deeper understanding of system dynamics under stochastic excitations.
  + Secondly, it identifies the interpretable form of the diffusion coefficient of generalized stochastic force, addressing the limitations of existing deterministic approaches.
  + The framework is applicable to both stochastic differential equations (SDEs) and stochastic partial differential equations (SPDEs).
  + Discovered systems show almost exact approximations to true system behavior and minimal relative error in derived equations of motion.
```

This repository contains the python codes of the paper 
  > + Tripura, T., Panda, S., Hazra, B., & Chakraborty, S. (2024). Data-driven discovery of interpretable Lagrangian of stochastically excited dynamical systems. [Paper](https://arxiv.org/abs/2402.17122)

![Discovery of Lagrangian Basis functions](figures/Basis.png)

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
