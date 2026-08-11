"""Paper-aligned QSSA Stiff-PINN for the ROBER problem in JAX ecosystem.

This script trains a small neural network to reproduce the solution of the
classic stiff ROBER chemical-kinetics problem (3 species, reaction rates
spanning ~9 orders of magnitude). Three ideas from three different papers
are combined here and also improved somehow:

1. QSSA reduction
   Ji, Weiqi, et al. "Stiff-pinn: Physics-informed neural network for stiff 
   chemical kinetics." The Journal of Physical Chemistry A 125.36 (2021): 8098-8106.
   Instead of asking the network to learn all three species y1, y2, y3, we
   assume species y2 is in "quasi-steady-state" (QSS) and solve for it with
   an algebraic formula instead of an ODE. This removes the fastest
   (stiffest) timescale from the network's job.

2. Hard initial-condition constraint
   Hao, Baoli, et al. "Stability in training PINNs for stiff PDEs: Why initial 
   conditions matter." arXiv preprint arXiv:2404.16189 (2024).
   The network never outputs y(t) directly. Instead we build
   y(t) = y0 + (bounded growth factor) * NN(t), so the initial condition
   y(0) = y0 is satisfied exactly, for any network output. I also used 
   log( t / t_max ) instead of t. This can half the training time.

3. GBCT residual scaling
   Yi, Yuxiao, et al. "An output scaling layer boosts deep neural networks
   for multiscale ODE systems." arXiv preprint arXiv:2512.05685 (2025).
   The physics residual dy/dt - rhs spans many orders of magnitude because
   ROBER is stiff, which is hard for a network to fit directly. We squash
   the residual with the Generalized Box-Cox Transformation (GBCT) before
   computing the loss, bringing huge and tiny residuals closer to the same
   scale.
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax


@dataclass(frozen=True)
class Config:
    # --- training hyper-parameters (how long / how big the network is) ---
    updates: int = 200_000
    n_points: int = 5_000       # time points sampled on a log grid over [t_min, t_max]
    batch_size: int = 32        # mini-batch size drawn from the time grid each step
    learning_rate: float = 1e-3
    width: int = 32             # neurons per hidden layer
    depth: int = 3              # number of hidden layers
    seed: int = 1

    # --- time domain: ROBER is usually solved over a huge log-time range ---
    t_min: float = 1e-6
    t_max: float = 1e+6

    # --- ROBER reaction rate constants, see stiff-PINNs.pdf Eq. 2.3/2.4 ---
    # k2/k1 ~ 1e9, which is exactly what makes this problem "stiff".
    k1: float = 0.04
    k2: float = 3e7
    k3: float = 1e4

    loss_scale: float = 1e2     # simple constant multiplier applied to the whole loss
    beta : float = 0.04         # controls how fast the hard-IC factor (1 - exp(-beta*t)) turns on
    lam  : float = 0.5

@dataclass(frozen=True)
class Result:
    model: object      # the trained equinox model
    loss: np.ndarray   # loss value recorded at every training step


def xavier_initializer(model, key):
    """Apply Xavier/Glorot uniform initialization to an eqx.nn.MLP."""

    initializer = jax.nn.initializers.xavier_uniform()

    # One independent random key per Linear layer
    layer_keys = jr.split(key, len(model.layers))

    for i, (layer, layer_key) in enumerate( zip(model.layers, layer_keys) ):
        
        new_weight = initializer( layer_key, layer.weight.shape, layer.weight.dtype )
        model = eqx.tree_at( lambda m: m.layers[i].weight, model, new_weight )

        zero_bias = jnp.zeros_like(layer.bias)
        model = eqx.tree_at( lambda m: m.layers[i].bias, model, zero_bias )

    return model


def gbct(x, lam=0.5, eps=1e-12):
    """Generalized Box-Cox Transformation. ( Yi, et al. 2025 )

    The paper defines GBCT as an odd (sign-preserving) power law:
        G(x) =  x**lam / lam        for x >= 0
        G(x) = -(-x)**lam / lam     for x <  0
    Small values get stretched up towards O(1) and large values get
    squashed down towards O(1) -- exactly what we need for the ROBER
    residual, which can be tiny or huge depending on where we are in time.

    Here we write it as ONE smooth formula instead of an if/else, so it
    stays differentiable everywhere (the `eps` term avoids a blow-up in the
    derivative at x = 0):
        gbct(x) = x * (x**2 + eps)**((lam - 1) / 2) / lam
    For x > 0 this is ~ x**lam / lam, and for x < 0 it is ~ -(-x)**lam / lam,
    which matches the paper's piecewise definition above.

    Note on how this is used below: in the original paper, G is applied to a
    regression *label* (a known target value). Here we don't have labels
    -- this is a physics-residual (PINN) loss, not supervised regression --
    so instead we apply G separately to the predicted dy/dt and to the
    physics right-hand-side, and compare G(dy/dt) to G(rhs). This keeps the
    same "squash multiscale magnitudes before computing the loss" idea from
    the paper, just adapted to a residual-based loss instead of a label.
    """
    return x * (x**2 + eps)**((lam - 1.0) / 2.0) / lam

 
def predict(model, time, config):
    """
    Bounded correction formulation:

        y(t) = y0  + ( 1 - exp(-βt) ) * NN( log(τ) )

    The network predicts a correction relative to y0.

    Hard initial-condition constraint: instead of training the network 
    to hit y(0) = y0 through an extra loss term, we bake the initial 
    condition directly into the  formula for y(t). 
    Since (1 - exp(-beta*t)) -> 0 as t -> 0, we get y(0) = y0 automatically.
    This removes a whole loss term and makes training far more stable for stiff
    problems. 

    The network only ever sees log(τ) as input (scaled by t_max first), not
    raw t. Using log-time as input spreads out the huge [1e-6, 1e6] second time
    range so the network can "see" both the fast and slow parts of the dynamics at once.

    Only y1 and y3 are predicted by the network. y2 (the fast/QSS species)
    is NOT predicted directly -- it is recovered below from the QSSA
    algebraic formula.
    """

    scaled_time = time / config.t_max
    network_input = jnp.array([ jnp.log( scaled_time ) ])

    # Numerically stable evaluation of:  (1 - exp(-βt))
    # (jnp.expm1 keeps precision when beta*time is small, unlike
    #  computing 1 - jnp.exp(-beta*time) directly)
    hard_factor = -jnp.expm1( - time * config.beta )

    initial_slow = jnp.array( [1.0, 0.0], dtype=time.dtype )  # y1(0) = 1, y3(0) = 0

    network_output = model(network_input)

    # Hard-constrained prediction of the two "slow" species y1 and y3
    y1, y3 = ( initial_slow + hard_factor * network_output )

    # QSSA algebraic solution for y2:
    #   y2 = ( -k3*y3 + sqrt( (k3*y3)^2 + 4*k1*k2*y1 ) ) / (2*k2)
    # The two lines below compute the exact same value, just rearranged
    # (multiplying by the conjugate) so we never subtract two nearly-equal
    # numbers -- this keeps y2 accurate even when y1, y3 are very small.
    root = jnp.sqrt( (config.k3 * y3) ** 2  + 4.0 * config.k1 * config.k2 * y1 )
    y2 = ( 2.0 * config.k1 * y1 / (config.k3 * y3 + root + 1e-30) )  # +1e-30 avoids 0/0

    return jnp.array([y1, y2, y3])  # full 3-species state, with y2 filled in via QSSA


def train(config):
    """Train only on the y1 and y3 residuals of the QSSA-reduced system.

    Because y2 is eliminated with the QSSA algebraic relation, we only
    need physics residuals for y1 and y3.
    """
    key, model_key = jr.split( jr.PRNGKey(config.seed) )

    # Network: takes time as input, outputs a 2-vector correction for [y1, y3]
    model = eqx.nn.MLP( in_size=1, out_size=2, width_size=config.width
                      , depth=config.depth, activation=jax.nn.tanh, key=model_key )

    model = xavier_initializer(model, model_key)

    # Sample time points on a log-spaced grid -> covers both the fast
    # (~1e-6 s) and slow (~1e6 s) parts of the ROBER dynamics evenly
    time_grid = jnp.geomspace(config.t_min, config.t_max, config.n_points)

    def loss_function(current_model, times):
        solution = lambda time: predict(current_model, time, config)

        y = jax.vmap(solution)(times)
        # dy/dt via automatic differentiation of the hard-constrained network
        dydt = jax.vmap(jax.jacfwd(solution))(times)
        y1, y2, y3 = y.T

        # Right-hand side of the ROBER ODEs, for y1 and y3 only (y2 already
        # comes from the QSSA formula in predict())
        rhs = jnp.stack([ -config.k1*y1 + config.k3*y2*y3, config.k2*y2**2 ], axis=1)

        dy_dt = dydt[:, jnp.array([0, 2])]

        # GBCT-transformed residual: instead of penalizing (dy/dt - rhs)
        # directly, we penalize G(dy/dt) - G(rhs). Both dy/dt
        # and rhs can be huge or tiny depending on where we are in time;
        # GBCT brings them onto a similar scale *before* squaring, which
        # makes the loss landscape much easier for the optimizer to follow.
        residual = gbct(dy_dt, config.lam) - gbct(rhs, config.lam)
        
        residual_loss = config.loss_scale * jnp.mean(residual**2)

        return ( residual_loss )
    
    # Learning-rate schedule: exponential decay down towards 10% of the
    # initial learning rate by the end of training
    learning_rate_schedule = optax.exponential_decay( init_value=config.learning_rate
                                                    , transition_steps=config.updates
                                                    , decay_rate=0.1, staircase=False )
    
    optimizer = optax.adam(learning_rate_schedule)
    # optimizer = optax.adam(config.learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # @eqx.filter_jit
    def train_step(current_model, current_state, times):
        loss, gradients = eqx.filter_value_and_grad(loss_function)(current_model, times)
        updates, current_state = optimizer.update( gradients, current_state, eqx.filter(current_model, eqx.is_array) )
        return eqx.apply_updates(current_model, updates), current_state, loss

    # losses = []
    # for _ in range(config.updates):
    #     key, batch_key = jr.split(key)
    #     indices = jr.randint(batch_key, (config.batch_size,), 0, config.n_points)
    #     times = time_grid[indices]
    #     model, optimizer_state, loss = train_step(model, optimizer_state, times)
    #     losses.append(loss)

    # Split the model into trainable arrays (params) vs everything else
    # (static) so the training loop below can be compiled with jax.lax.scan
    model_params, static_model = eqx.partition(model, eqx.is_array)

    def scan_step(carry, _):
        current_params, current_state, current_key = carry
        current_key, batch_key = jr.split(current_key)
        # batch_size - 2 random indices, plus the very first and very last
        # grid points added back in below -- so the earliest (fastest) and
        # latest (slowest) dynamics are always represented in every batch
        indices = jr.randint(batch_key, config.batch_size - 2, 0, config.n_points)
        times = jnp.concatenate( (time_grid[:1], time_grid[indices], time_grid[-1:]) )
        current_model = eqx.combine(current_params, static_model)

        current_model, current_state, loss = train_step( current_model, current_state, times )
        current_params, _ = eqx.partition(current_model, eqx.is_array)
        return (current_params, current_state, current_key), loss

    @eqx.filter_jit
    def run_training(current_params, current_state, current_key):
        # jax.lax.scan runs the whole training loop as one compiled op --
        # much faster than a plain Python for-loop for 100k update steps
        carry, losses = jax.lax.scan( scan_step, (current_params, current_state, current_key)
                                    , xs=None, length=config.updates, unroll=8 )
        return carry[0], carry[1], losses

    model_params, optimizer_state, losses = run_training( model_params, optimizer_state, key )
    model = eqx.combine(model_params, static_model)

    return Result(model=model, loss=np.asarray(jax.device_get(losses)))
