# voi-formation-control
Code reproducing simulations in the paper  "VoI-aware Scheduling Schemes for Multi-Agent Formation Control"

See also https://arxiv.org/abs/2507.06392

__________________________________________________________________________
Abstract: 

Formation control allows agents to maintain geometric patterns using local information, but most existing methods assume ideal communication. This paper introduces a goal-oriented framework combining control, cooperative positioning, and communication scheduling for first-order formation tracking. Each agent estimates its position using 6G network-based triangulation, and the scheduling of information updates is governed by Age of Information (AoI) and Value of Information (VoI) metrics. We design three lightweight, signaling-free scheduling policies and assess their impact on formation quality. Simulation results demonstrate the effectiveness of the proposed approach in maintaining accurate formations with no additional communication overhead, showing that worst-case formation adherence increases by 20%.

__________________________________________________________________________

MAIN: multi_runner.mat (launches Monte Carlo simulations);
To visualize results, use visualizer.m
