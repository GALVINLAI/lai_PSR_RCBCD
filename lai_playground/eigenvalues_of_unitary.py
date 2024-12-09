from lai_utils import generate_random_H_paulis
import jax
import numpy as np

n=3
prng_key=jax.random.PRNGKey(0)
H=generate_random_H_paulis(n, prng_key)
print(np.linalg.eig(H))