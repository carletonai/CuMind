# MuZero: DeepMind MCTS (`mctx`) Integration

## Objective  
Replace custom MCTS logic with `@google-deepmind/mctx.gumbel_muzero_policy`, a stable, optimized Monte Carlo Tree Search implementation used in MuZero-style agents. Designed for latent space search with learned dynamics.

---

## Library  
- **Name**: `@google-deepmind/mctx`  
- **Core function**: `gumbel_muzero_policy(...)`  
- **Backend**: JAX  
- **Batch support**: Yes  
- **Gumbel exploration**: Built-in

---

## Function Signatures

### `mctx.gumbel_muzero_policy(...)`

```python
output = mctx.gumbel_muzero_policy(
    params=...,
    rng_key=jax.random.PRNGKey(seed),
    root_fn=...,
    recurrent_fn=...,
    num_simulations=50,
    max_depth=5,
    qtransform=mctx.QTransform.FUNNEL,
)
```

---

## Required Components

### `root_fn(params, rng_key) → RootFnOutput`

Initial inference at the root node. Must return:

```python
mctx.RootFnOutput(
  prior_logits: Float[Array, "B A"],
  value: Float[Array, "B"],
  embedding: Float[Array, "B latent_dim"],
)
```

- `prior_logits`: raw action logits from the prediction head  
- `value`: scalar value estimate of the root  
- `embedding`: latent state output of representation model `h(obs)`

---

### `recurrent_fn(params, state, action) → RecurrentFnOutput`

Transition and prediction step. Must return:

```python
mctx.RecurrentFnOutput(
  reward: Float[Array, "B"],
  value: Float[Array, "B"],
  embedding: Float[Array, "B latent_dim"],
  prior_logits: Float[Array, "B A"],
)
```

- `reward`: predicted scalar reward from the transition  
- `value`: predicted value of the resulting state  
- `embedding`: next latent state from dynamics model  
- `prior_logits`: predicted action logits at the new state

---

## Outputs from `gumbel_muzero_policy`

```python
GumbelMuZeroPolicyOutput(
  action: Int[Array, "B"],
  logits: Float[Array, "B A"],
  root_value: Float[Array, "B"],
)
```

- `action`: action sampled from the improved search policy  
- `logits`: improved policy distribution from tree search  
- `root_value`: predicted value of the current state

---

## Integration Procedure

### 1. Define `root_fn`

```python
def root_fn(params, rng_key):
    embedding = model.representation(params, obs)
    logits, value = model.prediction(params, embedding)
    return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=embedding)
```

---

### 2. Define `recurrent_fn`

```python
def recurrent_fn(params, embedding, action):
    next_embedding, reward = model.dynamics(params, embedding, action)
    logits, value = model.prediction(params, next_embedding)
    return mctx.RecurrentFnOutput(
        reward=reward,
        value=value,
        embedding=next_embedding,
        prior_logits=logits,
    )
```

---

### 3. Replace existing MCTS logic with:

```python
mcts_output = mctx.gumbel_muzero_policy(
    params=model_params,
    rng_key=jax.random.PRNGKey(seed),
    root_fn=root_fn,
    recurrent_fn=recurrent_fn,
    num_simulations=50,
    max_depth=5,
)
```

Extract:
- `mcts_output.action`: action to execute  
- `mcts_output.logits`: improved policy (store for training)  
- `mcts_output.root_value`: value estimate for bootstrapping

---

## Training Notes

- Store `logits` from MCTS, not raw model policy, as training target  
- Use `root_value` for value loss or bootstrapped targets  
- Do not backpropagate through `mctx` — only through the model  
- `reward` and `value` should be scalars per sample  
- All inputs and outputs must be JAX arrays (`jax.numpy`)

---

## Debugging / Validation

- Ensure all tensor shapes match: `(B,)`, `(B, A)`, `(B, latent_dim)`  
- Start with 1–2 simulations and verify logit outputs  
- Validate value estimates are within expected scale  
- Confirm MCTS logits and selected actions vary with Gumbel noise

---

## Final Checklist

| Task | Complete |
|------|----------|
| `representation`, `dynamics`, and `prediction` defined | ☐ |
| `root_fn` returns logits, value, embedding | ☐ |
| `recurrent_fn` returns reward, value, next embedding, logits | ☐ |
| Custom MCTS fully replaced with `mctx.gumbel_muzero_policy` | ☐ |
| Replay buffer stores search logits and root value | ☐ |
| Environment compatibility validated | ☐ |

---

## References

- Library: https://github.com/google-deepmind/mctx  
- Docstring: `help(mctx.gumbel_muzero_policy)` in code  