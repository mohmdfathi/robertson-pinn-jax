# robertson-pinn-jax

Solving the stiff Robertson (ROBER) chemical kinetics problem with a Physics-Informed Neural Network.
The model is implemented in **JAX**, **Equinox**, and **Optax**, and combines Quasi-Steady-State-Assumption (QSSA) model reduction with Generalized Box-Cox residual scaling (GBCT) to make training stable despite the extremely stiff ROBER reaction rates (k2/k1 ~ 1e9). This project demonstrates automatic differentiation for ODE residuals, hard initial-condition constraints, multiscale residual rescaling, log-time sampling, and JIT-compiled training with `jax.lax.scan`.

The PINN uses a coordinate-based multilayer perceptron that maps log(τ), with τ = t / t_max, to a correction for the two slow species, y1 and y3. A hard initial-condition formulation, y(t) = y0 + (1 - exp(-βt)) · NN(log(τ)), ensures the initial condition is satisfied exactly, without an additional loss term. The fast species, y2, is never predicted directly — it is recovered from a Quasi-Steady-State-Assumption (QSSA) algebraic formula, which removes the stiffest timescale from the network's job. JAX automatic differentiation computes dy/dt for the ODE residual, and the residual is passed through a Generalized Box-Cox Transformation (GBCT, with a tunable exponent `lam`) before the loss is computed, so that residual values spanning many orders of magnitude are brought closer to the same scale. The model is trained by minimizing this rescaled residual with Equinox and the Optax Adam optimizer, using an exponentially decaying learning rate and a JIT-compiled `jax.lax.scan` training loop.

The PINN closely reproduces the reference ROBER trajectory across the full log-time domain, from the fast initial transient to the slow relaxation phase, despite reaction rates that differ by nine orders of magnitude.

![ROBER results](rober_comparison.png)
 

**References**

- Ji, Weiqi, et al. "Stiff-pinn: Physics-informed neural network for stiff chemical kinetics." *The Journal of Physical Chemistry A* 125.36 (2021): 8098-8106.
- Hao, Baoli, et al. "Stability in training PINNs for stiff PDEs: Why initial conditions matter." *arXiv preprint* arXiv:2404.16189 (2024).
- Yi, Yuxiao, et al. "An output scaling layer boosts deep neural networks for multiscale ODE systems." *arXiv preprint* arXiv:2512.05685 (2025).
