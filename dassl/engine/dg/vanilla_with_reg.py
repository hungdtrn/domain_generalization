from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy


@TRAINER_REGISTRY.register()
class VanillaWithReg(TrainerX):
    """Vanilla model.
    
    A.k.a. Empirical Risk Minimization, or ERM.
    """

    def forward_backward(self, batch):
        input, target = self.parse_batch_train(batch)
        output, (_, (tokens, feature)) = self.model(input, return_feature=True)
        
        reg_loss = F.mse_loss(feature, tokens)
        
        print(reg_loss)
        
        loss = F.cross_entropy(output, target) + reg_loss
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
        input = input.to(self.device)
        target = target.to(self.device)
        return input, target
