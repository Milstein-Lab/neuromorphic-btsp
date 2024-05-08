# Neuromorphic one-shot learning utilizing a phase-transition material

This repository contains code and data to reproduce the results in _Galloni et al. (2024)_ 
"**Neuromorphic one-shot learning utilizing a phase-transition material**". The full final version of this paper is available at 
[PNAS](https://www.pnas.org/doi/10.1073/pnas.2318362121).

An earlier version of the paper is also available as a preprint on [arXiv](https://arxiv.org/abs/2310.00066):
"**Temporal credit assignment for one-shot learning utilizing a phase transition material**".

In this paper, we simulated the phase transition properties of vanadium oxide (VO2) and used them to perform neuroscience-inspired one-shot learning. 
We demonstrate multiple possible uses of VO2 by integrating these devices into analog electronic circuits for emulating both spiking neurons with variable refractory periods, as well as slow eligibility traces that are critical for behavioral timescale synaptic plasticity (BTSP) — a learning rule inspired directly from experimental observations in pyramidal neurons of the hippocampus. We further show how this learning rule can be used for biologically plausibile reinforcement learning to navigate in 2D environments using the Successor Representation.


## Citation
If you use this codebase, or otherwise found our work valuable, please cite the following source:
```
@article{galloni2024btsp,
        title   = {Neuromorphic one-shot learning utilizing a phase-transition material},
        author  = {Galloni, Alessandro R. and Yuan, Yifan and Zhu, Minning and Yu, Haoming 
                    and Bisht, Ravindra S. and Wu, Chung-Tse Michael and Grienberger, Christine 
                    and Ramanathan, Shriram and Milstein, Aaron D.},
        journal = {Proceedings of the National Academy of Sciences},
        volume  = {121},
        number  = {17},
        pages   = {e2318362121},
        year    = {2024},
        url     = {https://www.pnas.org/doi/10.1073/pnas.2318362121},
        doi     = {10.1073/pnas.2318362121},
}
```