import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy


@TRAINER_REGISTRY.register()
class VanillaWithDomain(TrainerX):
    """Vanilla model.
    
    A.k.a. Empirical Risk Minimization, or ERM.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_domains = 4

    def forward_backward(self, batch):
        input, domain, target = self.parse_batch_train(batch)
        output = self.model((input, domain))
        loss = F.cross_entropy(output, target)
        self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, target)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        target = batch["label"]
        domain = batch["domain"]

        input = input.to(self.device)
        target = target.to(self.device)
        domain = domain.to(self.device)

        return input, domain, target
    
    def parse_batch_test(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = None

        input = input.to(self.device)
        label = label.to(self.device)

        return (input, domain), label