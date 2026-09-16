#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate ascend
source $CONDA_PREFIX/Ascend/ascend-toolkit/set_env.sh
source $CONDA_PREFIX/Ascend/nnal/atb/set_env.sh
export ASCEND_SIMULATOR_MODE=1
export CMAKE_ASC_RUN_MODE=cpu
export SOC_VERSION=Ascend910B1
python hf_e2e_llm_sim.py
