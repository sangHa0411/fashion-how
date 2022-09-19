
import torch

class LinearWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """
    Linear Warmup Schedule 방식으로 learning rate을 조정하기 위한 클래스입니다.
    """
    def __init__(self, optimizer, total_steps, warmup_steps, last_epoch=-1):

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(
                0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
            )

        super(LinearWarmupScheduler, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch) 