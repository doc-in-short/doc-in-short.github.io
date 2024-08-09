---
layout: doc
title: Catboost shap values
---

Calculate and plot catboost shap values.

```python
import shap
shap.initjs()

def try_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0

sample = data.sample(n=1000).reset_index(drop=True)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(sample.features)

shap.summary_plot(
    shap_values,
    feature_names=features,
    features=np.array(sample.features.apply(lambda x: list(map(try_float, x))).to_list()),
)
```

Result example:

![](/assets/images/catboost_shap.png)
