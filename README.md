## Research Background

During our heat-transfer research, we used both MATLAB and
Python/PyTorch to develop and evaluate neural-network surrogates
for temperature prediction.

The work began with an analytical single-fin study: generating
temperature profiles from a heat-transfer equation and training
neural networks to predict normalized temperature from position
and fin length.

The research then extended to 3D heat-sink temperature prediction
using OpenFOAM CFD data and PyTorch, as described in our
ASME FEDSM 2026 paper.

This repository presents the MATLAB implementation of the
analytical fin study, including data generation, neural-network
training, validation, and visualization.
