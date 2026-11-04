"""Train the from-scratch MLP on XOR.

XOR is the smallest task that a single neuron cannot solve, so it is a useful
proof that the hidden layers, the backpropagation implementation, and the
optimizer all work together. Run from the repository root:

    python -m examples.train_xor
"""

from pathlib import Path

from nn.losses import BinaryCrossEntropyWithLogits
from nn.model import MLP
from nn.optimizers import Adam
from nn.serialization import save_model
from training.metrics import binary_metrics
from training.train import train

X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
Y = [0.0, 1.0, 1.0, 0.0]


def main() -> None:
    model = MLP([2, 8, 8, 1], hidden_activation="relu", seed=7)
    loss_fn = BinaryCrossEntropyWithLogits()
    optimizer = Adam(lr=0.05)

    print(model.describe())
    history = train(
        model,
        X,
        Y,
        loss_fn,
        optimizer,
        epochs=3000,
        batch_size=4,
        shuffle=True,
        seed=7,
        verbose=True,
        log_every=500,
    )

    scores = [model.predict_proba(x) for x in X]
    metrics = binary_metrics([int(y) for y in Y], scores)

    print("\n  x        target   P(class=1)   prediction")
    for inputs, target, score in zip(X, Y, scores):
        print(
            f"  {inputs}    {int(target)}       {score:8.5f}     "
            f"{int(score >= 0.5)}"
        )

    print(
        f"\nfinal loss={history.loss[-1]:.8f}  "
        f"accuracy={metrics['accuracy']:.2f}"
    )

    output = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "results"
        / "xor_mlp.json"
    )
    save_model(
        model,
        output,
        metadata={
            "task": "xor",
            "dataset": "xor-4-samples",
            "optimizer": "adam",
            "learning_rate": 0.05,
            "epochs": 3000,
            "batch_size": 4,
            "seed": 7,
            "final_loss": history.loss[-1],
            "accuracy": metrics["accuracy"],
        },
    )
    print(f"model saved to {output}")


if __name__ == "__main__":
    main()
