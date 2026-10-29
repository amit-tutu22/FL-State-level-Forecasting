# APFL in this experiment

Adaptive Personalized Federated Learning (Deng, Kamani and Mahdavi, 2020) keeps a global model and a local model at each client. The model that is scored is the mixture

```
theta = alpha * local + (1 - alpha) * global
```

with `alpha` in `[0, 1]`. `alpha = 0` is the global model. `alpha = 1` is the client's own model.

## What this code does

At each rolling origin, every state starts from the same global weights and from `alpha = 0.5`.

1. The client fits its local weights with full-batch Adam on its fit windows. The loss is mean squared error. There is no FedProx penalty in this method.
2. `alpha` takes one gradient step on that client's six-month holdout loss. The step size is chosen on the validation origins and then frozen. `alpha` is clipped back into `[0, 1]`.
3. The client is scored with the mixture above. The server averages the local weights, weighted by the number of fit windows. It does not average the mixtures.
4. The communication round that is kept is the one with the lowest equal-state holdout RMSE. The origin month is not part of that choice.

## Where this differs from the paper

The paper studies minibatch classification. This experiment uses a full-batch GRU regression on monthly tax windows, early-stops the round on a time holdout, and takes one mixing-weight step per round rather than a step on every minibatch. Those are implementation choices for this protocol, not a claim that the paper's optimizer was reproduced line for line.
