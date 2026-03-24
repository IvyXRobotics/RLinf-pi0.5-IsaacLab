# AMP training entry-point for RLinf (pi0.5 variant).
# Identical to train_embodied_agent.py except it uses AMPEmbodiedFSDPActor.
#
# Usage (inside Apptainer container):
#   cd /workspace/RLinf && python examples/embodiment/train_embodied_agent_amp.py \
#       --config-path /workspace/RLinf/examples/embodiment/config/ \
#       --config-name isaaclab_franka_stack_cube_amp_ppo_openpi_pi05 \
#       rollout.model.model_path=/workspace/RLinf/outputs/RLinf-pi05-SFT-Stack-cube \
#       actor.model.model_path=/workspace/RLinf/outputs/RLinf-pi05-SFT-Stack-cube \
#       amp.hdf5_path=/workspace/RLinf/datasets/rendered.hdf5

import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.amp_fsdp_actor_worker import AMPEmbodiedFSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="isaaclab_franka_stack_cube_amp_ppo_openpi_pi05",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_placement = component_placement.get_strategy("actor")
    actor_group = AMPEmbodiedFSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )

    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
