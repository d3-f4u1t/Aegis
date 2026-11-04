import pytest

from nn.losses import BinaryCrossEntropyWithLogits
from nn.model import MLP
from nn.optimizers import Adam
from training.train import train

XOR_X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
XOR_Y = [0.0, 1.0, 1.0, 0.0]


def test_xor_converges():
    """The classic non-linearly-separable toy task.

    If a hand-written MLP can separate XOR, forward propagation, backpropagation,
    and the optimizer are all working together.
    """
    model = MLP([2, 8, 8, 1], hidden_activation="relu", seed=7)
    optimizer = Adam(lr=0.05)
    history = train(
        model,
        XOR_X,
        XOR_Y,
        BinaryCrossEntropyWithLogits(),
        optimizer,
        epochs=3000,
        batch_size=4,
        seed=7,
    )

    assert history.loss[-1] < 0.02, f"final loss was {history.loss[-1]:.6f}"
    probabilities = [model.predict_proba(x) for x in XOR_X]
    assert probabilities[0] < 0.1
    assert probabilities[1] > 0.9
    assert probabilities[2] > 0.9
    assert probabilities[3] < 0.1


def test_same_seed_gives_identical_initialisation():
    first = MLP([3, 5, 1], seed=42)
    second = MLP([3, 5, 1], seed=42)
    assert [p.value for p in first.parameters()] == [
        p.value for p in second.parameters()
    ]


def test_different_seeds_give_different_initialisation():
    first = MLP([3, 5, 1], seed=42)
    second = MLP([3, 5, 1], seed=43)
    assert [p.value for p in first.parameters()] != [
        p.value for p in second.parameters()
    ]


def test_training_history_has_one_entry_per_epoch():
    model = MLP([2, 4, 1], seed=1)
    history = train(
        model,
        XOR_X,
        XOR_Y,
        BinaryCrossEntropyWithLogits(),
        Adam(lr=0.01),
        epochs=5,
        batch_size=2,
        seed=1,
    )
    assert len(history.loss) == 5


def test_mismatched_input_and_target_lengths_are_rejected():
    model = MLP([2, 2, 1], seed=1)
    with pytest.raises(ValueError, match="same length"):
        train(
            model,
            XOR_X,
            [0.0, 1.0],
            BinaryCrossEntropyWithLogits(),
            Adam(),
            epochs=1,
        )
