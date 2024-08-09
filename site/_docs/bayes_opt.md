---
layout: doc
title: Bayesian optimization
---

Optimize given target function with bayes_opt package.

```python
from bayes_opt import BayesianOptimization

def target(x):
    return -(x - 2) ** 2


optimizer = BayesianOptimization(
    f=target,
    pbounds={
        'x': (1, 5),
    },
    verbose=2,
    random_state=0,
)
optimizer.set_gp_params(alpha=0.001)

optimizer.maximize(
    init_points=3,
    n_iter=4,
)
```
