---
layout: doc
title: Catboost feature importance
---

Plots top feature importances for catboost model.

```python
plt.figure(figsize=(6, 7))

order = np.argsort(clf.feature_importances_)[::-1]
order = order[:15]

sns.barplot(
    x=clf.feature_importances_[order],
    y=np.array(clf.feature_names_)[order],
)

plt.title('Top {} features for prediction'.format(order.shape[0]))

plt.show()
```

Result example:

![](/assets/images/catboost_fstr.png)
