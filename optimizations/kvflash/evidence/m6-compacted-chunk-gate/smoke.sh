#!/bin/bash
# Run under: flock /mnt/kimi-k3/build/3090.lock /mnt/kimi-k3/build/m6c-ab/smoke.sh <name> <bin> [TEST_COMPACT]
apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [ -n "$apps" ]; then echo "GPU busy: $apps"; exit 1; fi
export M6_PROMPTS=${M6_PROMPTS:-2} M6_TOKENS=${M6_TOKENS:-12000}
if [ -n "$3" ]; then export M6_TEST_COMPACT=$3; fi
/mnt/kimi-k3/build/m6c-ab/ab_run.sh "$1" "$2"
