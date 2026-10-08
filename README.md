## Separate PINN Extension

Separately from the published research, we extended this MATLAB
project with a physics-informed neural network (PINN).
This extension was not included in the ASME FEDSM 2026 paper.

The baseline neural network learns from temperature data alone.
The PINN also includes penalties for violating the governing
fin heat-transfer equation and the base and tip boundary conditions.

Both models were evaluated on the same held-out fin lengths.

### Test Results

| Metric | Data-only neural network baseline | PINN |
|---|---:|---:|
| MAE | 0.009140 | 0.008527 |
| RMSE | 0.011525 | 0.010419 |

In this experiment, the PINN reduced test MAE by 6.7% and
test RMSE by 9.6%. These errors measure the normalized
temperature ratio, not temperature in degrees.

Boundary conditions are imposed through loss penalties,
so they are not guaranteed to be satisfied exactly.

### PINN Predictions vs Baseline and Theory

![PINN comparison](pinn_vs_baseline.png)

### PINN Training Loss

![PINN training loss](pinn_training_loss.png)

### Running the PINN Extension

1. Run `generate_fin_data.m` to generate the analytical dataset.
2. Run `train_fin_surrogate.m` to train and save the baseline.
3. Run `train_fin_pinn.m` to train the PINN and compare results.

The PINN script saves `fin_pinn_model.mat`,
`pinn_training_loss.png`, and `pinn_vs_baseline.png`.
