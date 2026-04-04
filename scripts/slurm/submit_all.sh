#!/bin/bash
# Submit all 5 lerp experiments as separate SLURM jobs.
# Run this from the cluster login node (NOT inside apptainer):
#   bash scripts/slurm/submit_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "$SCRIPT_DIR/job_lerp00.sh"
sbatch "$SCRIPT_DIR/job_lerp02.sh"
sbatch "$SCRIPT_DIR/job_lerp05.sh"
sbatch "$SCRIPT_DIR/job_lerp08.sh"
sbatch "$SCRIPT_DIR/job_lerp10.sh"

echo "Submitted 5 jobs. Check status with: squeue -u \$USER"
