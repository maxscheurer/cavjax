import jax

# Tests opt in explicitly; importing cavjax itself never changes global JAX state.
jax.config.update("jax_enable_x64", True)
