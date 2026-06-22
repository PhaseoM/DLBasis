from .config import conf


def warmup_linear_decay(warmup_steps, total_steps):

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return max(0, (total_steps - step) / (total_steps - warmup_steps))

    return lr_lambda
