# AMP-augmented actor worker for RLinf.
#
# Wraps EmbodiedFSDPActor with AMPActorMixin so the discriminator is trained
# alongside the PPO policy.  No changes to the base actor are required.

from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf_amp.amp_actor_mixin import AMPActorMixin


class AMPEmbodiedFSDPActor(AMPActorMixin, EmbodiedFSDPActor):
    """EmbodiedFSDPActor extended with AMP discriminator training.

    The MRO ensures AMPActorMixin's overrides of
    ``compute_advantages_and_returns`` and ``run_training`` take effect
    before EmbodiedFSDPActor's implementations.

    Requires ``amp:`` section in the training yaml (see AMPActorMixin docstring).
    """

    def init_worker(self) -> None:
        # Run base FSDP setup first (model, optimizer, offload, rank wiring)
        super().init_worker()
        # Then initialise AMP components (discriminator, replay buffer, dataset)
        self.init_amp()

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        # Save pi0.5 model + optimizer (base class)
        super().save_checkpoint(save_path, step)
        # Save discriminator + normalizer alongside the model checkpoint
        self.save_amp_state(save_path)

    def load_checkpoint(self, load_path: str) -> None:
        # Restore pi0.5 model + optimizer (base class)
        super().load_checkpoint(load_path)
        # Restore discriminator + normalizer
        self.load_amp_state(load_path)