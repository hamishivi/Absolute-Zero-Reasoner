set -x

python mason.py \
    --cluster ai2/jupiter \
    --gpus 4 \
    --budget ai2/oe-adapt \
    --priority urgent \
    --workspace ai2/olmo-instruct \
    --description "AZR 7B conditioning dataset training" \
    --image ai2/cuda12.8-dev-ubuntu22.04-notorch \
    --pure_docker_mode \
    -- \
    bash scripts/selfplay/7b_conditioning.sh
