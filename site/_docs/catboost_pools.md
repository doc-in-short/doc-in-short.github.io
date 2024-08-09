---
layout: doc
title: Catboost pools
---

### 1. Describe features

{% include code-header.html %}
```python
float_features = [
    '',
]

cat_features = [
    '',
]

text_features = [
    '',
]

features = float_features + cat_features + text_features
ranker_features = float_features + cat_features
```

### 2. Describe pools

```python
train_pool = catboost.Pool(
    data=train[features],
    label=train['target'],
    cat_features=cat_features,
#     text_features=text_features,
)
val_pool = catboost.Pool(
    data=val[features],
    label=val['target'],
    cat_features=cat_features,
#     text_features=text_features,
)

clf = catboost.CatBoostClassifier(
    eval_metric='AUC',
    n_estimators=1000,
)

clf.fit(
    X=train_pool,
    eval_set=val_pool,
    verbose=10,
    early_stopping_rounds=100,
)
```
