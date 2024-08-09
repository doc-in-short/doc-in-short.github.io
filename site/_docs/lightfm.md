---
layout: doc
title: LightFM
---

Lighfm fitting script:

```python
from scipy import sparse


train['user_id'] = preprocessing.LabelEncoder().fit_transform(train['user'])
train['item_id'] = preprocessing.LabelEncoder().fit_transform(train['item'])

sm_train = sparse.coo_matrix(
    (
        np.ones(train.shape[0]),
        [
            np.array(train.user_id),
            np.array(train.item_id),
        ],
    ),
    shape=(train.user_id.max() + 1, train.item_id.max() + 1),
)

model = lightfm.LightFM(loss='warp')
# %time model.fit(sm_train, epochs=30, num_threads=8)
```

Iterative fitting:

```python
from IPython import display
from lightfm import evaluation
import pickle

model = lightfm.LightFM(loss='warp')
history = collections.defaultdict(list)

N_EPOCH = 40

best_score = None

for epoch in tqdm.tqdm(range(N_EPOCH)):
    print('Fitting...')
    model.fit_partial(sm_train, epochs=25, num_threads=12)

    print('Validating...')
    history['at1000'].append(
        evaluation.precision_at_k(model, sm_val, k=1000, num_threads=12).mean()
    )
    history['at12'].append(
        evaluation.precision_at_k(model, sm_val, k=12, num_threads=12).mean()
    )

    display.clear_output()
    plt.figure(figsize=(15, 7))

    plt.subplot(121)
    plt.plot(history['at1000'])
    plt.title('Precision@1000')

    plt.subplot(122)
    plt.plot(history['at12'])
    plt.title('Precision@12')

    plt.show()

    if best_score is None or best_score < history['at1000'][-1]:
        print('New best score!')
        best_score = history['at1000'][-1]

        with open('lightfm.model', 'wb') as f:
            pickle.dump(model, f)
```
