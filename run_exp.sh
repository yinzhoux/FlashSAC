#!/bin/bash

run_in_tmux() {
    local SESSION_NAME=$1
    local COMMAND=$2
    local LOG_DIR="./logs"
    mkdir -p "$LOG_DIR"
    local TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    local LOG_FILE="${LOG_DIR}/${SESSION_NAME}_${TIMESTAMP}.log"

    tmux has-session -t "$SESSION_NAME" 2>/dev/null
    if [ $? -eq 0 ]; then
        tmux kill-session -t "$SESSION_NAME"
    fi

    tmux new-session -d -s "$SESSION_NAME" "$COMMAND > $LOG_FILE 2>&1"
    
    echo "成功在会话 [$SESSION_NAME] 中启动任务，日志查看: tail -f $LOG_FILE"
}

# 1. sac normal training with:
#    Reward norm; Encoder weight norm;
# run_in_tmux "1-sac_normal" \
# "export MUJOCO_GL=egl && \
# uv run python ~/zjx/FlashSAC/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-sac" \
#                 --overrides exp_name=sac \
#                 --overrides group_name=sac

# 2. metra normal training with:
#    Reward norm; NO Encoder weight norm; episode 200
# run_in_tmux "metra_normal" \
# "export MUJOCO_GL=egl && \
# uv run python ~/zjx/FlashSAC/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra \
#                 --overrides exp_name=metra-normal"

# 3. rerun 2 for episode 1000.

# 4. remove reward normalization
#    NO Reward norm; NO Encoder weight norm; episode 1000
# run_in_tmux "4-metra-no-rewd-norm" \
# "export MUJOCO_GL=egl && \
# uv run python ~/zjx/FlashSAC/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-no-rewd-norm \
#                 --overrides exp_name=metra-normal-no-rewd-norm"


# 5. use gaussian encoder
#    NO Reward norm; NO Encoder weight norm; episode 1000
run_in_tmux "5-gaussian" \
"export MUJOCO_GL=egl && \
uv run python ~/zjx/FlashSAC/train.py \
                --config_name metra_base \
                --overrides agent=metra-no-rewd-norm \
                --overrides exp_name=metra-gaussian"