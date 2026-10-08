#!/bin/bash
# Build the graph-launch-cost microbench on lucebox4.
# Run on the box from ~/qwen4exp-launchcost/ (synced there, not /tmp).
set -euo pipefail
cd ~/qwen4exp-launchcost
/opt/rocm/bin/hipcc -O3 -DNDEBUG --offload-arch=gfx1151 -o graph_launch_cost graph_launch_cost.hip
echo "built: $(file graph_launch_cost)"
