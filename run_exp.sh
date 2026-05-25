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

root_dir=/home/helix/projects/METRA-Projects/FlashSAC

# test running
# export MUJOCO_GL=egl
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base_test \
#                 --overrides agent=metra-test \
#                 --overrides exp_name=env_info

# 1. sac normal training with:
#    Reward norm; Encoder weight norm;
# run_in_tmux "1-sac_normal" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-sac" \
#                 --overrides exp_name=sac

# 2. metra normal training with:
# 12. rerun this after refactor dual lambda update.

#    Reward norm; NO Encoder weight norm; episode 1000
# run_in_tmux "2-metra_normal" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra \
#                 --overrides exp_name=metra-normal"

# 3. rerun 2 for episode 1000.

# 4. remove reward normalization
# 11. rerun this after refactor dual lambda update.

#    NO Reward norm; NO Encoder weight norm; episode 1000
# run_in_tmux "4-metra-no-rewd-norm" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-no-rewd-norm \
#                 --overrides exp_name=metra-normal-no-rewd-norm"


# 5. use gaussian encoder
#    NO Reward norm; NO Encoder weight norm; episode 1000
# run_in_tmux "5-gaussian" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-gaussian \
#                 --overrides exp_name=metra-gaussian"

# 6. use simple encoder
#    NO Reward norm; NO Encoder weight norm; episode 1000
# run_in_tmux "6-simple" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-simple \
#                 --overrides exp_name=metra-simple"

# 7. metra normal training with:
#    Reward norm; NO Encoder weight norm; episode 1000, longer train
# run_in_tmux "7-metra_normal-long" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base_long \
#                 --overrides agent=metra \
#                 --overrides exp_name=metra-normal-long"

# 8. discrete skills: 4
#    NO Reward norm; NO Encoder weight norm; episode 1000
# run_in_tmux "8-discrete" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-discrete \
#                 --overrides exp_name=metra-discrete"

# 9. add encoder weight norm: 4
# 13. rerun this after refactor lambda update.

#    NO Reward norm; weight norm; episode 1000
# run_in_tmux "9-encoder-weight-norm" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-encoder-weight-norm \
#                 --overrides exp_name=metra-encoder-weight-norm"

# 10. larger discrete skill space: 8
#    NO Reward norm; NO Encoder weight norm; episode 1000
# run_in_tmux "10-discrete-larger" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-discrete-larger \
#                 --overrides exp_name=metra-discrete-larger"

# 14. metra normal training with low init lambda:

#    Reward norm; Encoder weight norm; episode 1000
# run_in_tmux "14-metra-low-lambda-init" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-low-lambda-init \
#                 --overrides exp_name=metra-normal-low-lambda"

# 15. metra normal on cpu.

#    Reward norm; NO Encoder weight norm; episode 1000
# run_in_tmux "15-metra_normal-cpu" \
# "export MUJOCO_GL=egl && \
# uv run python ${root_dir}/train.py \
#                 --config_name metra_base \
#                 --overrides agent=metra-cpu \
#                 --overrides exp_name=metra-normal_cpu" \
#                 --overrides 

#-===========METRA train===============
# run_in_tmux "2-metra-test-position-info-param-tuning" \
# "export MUJOCO_GL=egl && \
# uv run python METRA_train.py \
#   --config-path configs \
#   --config-name metra_aligned_base \
#   env.env_name=Ant-v4 \
#   group_name=compare \
#   exp_name=ant_aligned \
#   seed=0  \
#   exp_name=test-postion-info"

# test
# run_in_tmux "2-metra-test-position-info-test" \
# "export MUJOCO_GL=egl && \
# uv run python METRA_train.py \
#   --config-path configs \
#   --config-name metra_aligned_base \
#   env.env_name=Ant-v4 \
#   group_name=compare \
#   exp_name=test-info \
#   seed=0  \
#   exp_name=test-postion-info \
#   n_epochs=1 "

#-===========METRA train===============
run_in_tmux "3-metra-p-gau" \
"export MUJOCO_GL=egl && \
uv run python METRA_train.py \
  --config-path configs \
  --config-name metra_aligned_base \
  env.env_name=Ant-v4 \
  group_name=compare \
  exp_name=ant_aligned_gau \
  seed=0  \
  exp_name=test-postion-info \
  agent=metra_aligned-gau"

# #-===========METRA train===============
# run_in_tmux "4-metra-p-episode200" \
# "export MUJOCO_GL=egl && \
# uv run python METRA_train.py \
#   --config-path configs \
#   --config-name metra_aligned_base \
#   env.env_name=Ant-v4 \
#   group_name=compare \
#   exp_name=ant_aligned \
#   seed=0  \
#   exp_name=test-postion-info \
#   agent_name=metra_aligned-episode200 \
#   env.max_episode_steps=200"
