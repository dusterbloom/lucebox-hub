# Environment for R1(e) on lucebox6 (sourced on the host). Nothing under /tmp.
export R1E=$HOME/r1e
export TMPDIR=$R1E/tmp HF_HOME=$R1E/hf PIP_CACHE_DIR=$R1E/pipcache TRITON_CACHE_DIR=$R1E/tmp/triton
export HIP_VISIBLE_DEVICES=1
# no python3-dev on the host: headers extracted from the libpython3.12-dev deb
export CPATH=$R1E/debs/x/usr/include:$R1E/debs/x/usr/include/python3.12:$R1E/debs/x/usr/include/x86_64-linux-gnu/python3.12
export PYTHONPATH=$R1E/code/repo:$R1E/code/am_qwen35:$R1E/code/r1e
