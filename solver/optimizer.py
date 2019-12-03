import texar.torch as tx


def build_optimizer(model):
    static_lr = 1e-5
    # 设置优化器
    vars_with_decay = []
    vars_without_decay = []

    for name, param in model.named_parameters():
        # 对于 layer_norm 和 bias 不需要学习速率衰减
        if 'layer_norm' in name or name.endswith('bias'):
            vars_without_decay.append(param)
        else:
            vars_with_decay.append(param)

    # 优化器参数
    opt_params = [{
        # 对于需要衰减的参数
        'params': vars_with_decay,
        'weight_decay': 0.01,
    }, {
        'params': vars_without_decay,
        'weight_decay': 0.0,
    }]

    # BERT adam 优化器
    optimzier = tx.core.BertAdam(
        opt_params, betas=(0.9, 0.999), eps=1e-8, lr=static_lr)

    return optimzier
