# DISCOTRESS tools

This repository contains various code to analyse the output of the [DISCOTRESS](https://github.com/danieljsharpe/DISCOTRESS) software for the simulation and analysis of Markov chains.

## Overview of analysis scripts

- *traj\_analysis* contains scripts to process the trajectory and path statistics files output by DISCOTRESS. The scripts can be used to compute the mean and variance of the first passage time (FPT) distribution, plot the complete FPT distribution, (and analyse the distributions of other dynamical quantities in the first passage path ensemble (FPPE), such as the path action and entropy flow), and plot the time-dependent occupation probability distributions for macrostates of the network.

- *discotress\_coarsegrainer* contains scripts to process the output of the **WRAPPER DIMREDN** (i.e. dimensionality reduction) functionality of the DISCOTRESS program. This mode of DISCOTRESS simulates sets of many short nonequilibrium trajectories initialised from each of the macrostates. The analysis scripts here can be used to subsequently estimate a coarse-grained Markov chain from the simulation data (using maximum-likelihood or Gibbs sampling methods) and validate the estimated coarse-grained Markov chain (by implied timescale, Chapman-Kolmogorov, and correlation function tests).

- *markov\_chain\_analysis* contains code to read in the network topology files in DISCOTRESS format and analyse various features of the dynamics of Markov chains at a microscopic level of detail. Features include variational optimisation of an initial partitioning of a Markov chain based on the local equilibrium approximation for the coarse-grained Markov chain, computation of the visitation probabilities for nodes, isocommittor cut analysis to characterise the productive probability flow through dynamical bottlenecks, and surprisal analysis to measure the difference between two Markov chains with the same topology.

## Miscellaneous scripts

- *ctmc_to_dtmc.py* converts the DISCOTRESS input files for a continuous-time Markov chain to a corresponding discrete-time Markov chain.
