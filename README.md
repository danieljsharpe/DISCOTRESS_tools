# DISCOTRESS_tools

This repository contains various code to analyse transition networks (Markov chains).

In the folder *discotress_coarsegrainer*, there are scripts to process the output of the DISCOTRESS program for dynamical simulation of Markov chains (see the DISCOTRESS repo, github.com/danieljsharpe/DISOCTRESS, for more information), and to subsequently estimate a coarse-grained Markov chain from the simulation data (using maximum-likelihood or Gibbs sampling methods) and validate the estimated coarse-grained Markov chain (by implied timescale, Chapman-Kolmogorov, and correlation function tests).

In the folder *markov_chain_analysis*, there are scripts to analyse the dynamics on Markov chains (including variational optimisation of an initial partitioning of a Markov chain using the local equilibrium approximation for the coarse-grained Markov chain, isocommittor cut analysis for dynamical bottlenecks, computation of mean first passage times, and surprisal analysis to measure the difference between two Markov chains with the same topology).
