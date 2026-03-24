# rlinf_amp — AMP (Adversarial Motion Priors) adaptation for RLinf + stack-cube.
#
# Components
# ----------
# hdf5_motion_dataset : HDF5MotionDataset  — loads rendered.hdf5 directly (no .npz)
# amp_discriminator   : AMPDiscriminator   — standalone style-reward MLP
# amp_replay_buffer   : AMPReplayBuffer    — ring buffer for policy AMP transitions
# amp_normalizer      : AMPNormalizer      — running mean/std for AMP obs
# amp_actor_mixin     : AMPActorMixin      — adds discriminator training to FSDP actor
