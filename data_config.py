
train_path = 'data/train_set.csv'
valid_path = 'data/valid_set.csv'

train_batch_size = 6
display_steps = 100
warmup_proportion = 0.1
optimizer = {
    "warmup_steps": {"start": 10000, "end": 20000, "dtype": int},
    "static_lr": {"start": 1e-3, "end": 1e-2, "dtype": float}
}

train_hparams = {
    'batch_size': train_batch_size,
    'num_epochs': 2,
    'shuffle': True
}

model_hparams = {
    "num_classes": 1,
    # "logit_layer_kwargs": None,
    "clas_strategy": "cls_time",
    # "max_seq_length": None,
    "dropout": 0.1,
    "name": "bert_classifier"
}


def get_lr_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    r"""Calculate the learning rate multiplier given current step and the number
    of warm-up steps. The learning rate schedule follows a linear warm-up and
    linear decay.
    """
    step = min(step, total_steps)

    multiplier = (1 - (step - warmup_steps) / (total_steps - warmup_steps))

    if warmup_steps > 0 and step < warmup_steps:
        warmup_percent_done = step / warmup_steps
        multiplier = warmup_percent_done

    return multiplier
