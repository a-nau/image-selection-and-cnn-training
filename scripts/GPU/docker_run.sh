#!/bin/bash

session_name=parcel2d
# kill old session if exists
tmux kill-session -t "${session_name}"
sleep 1

# start new tmux session
tmux new-session -s "${session_name}" 'nvidia-docker run -it --rm --name parcel2d \
-e NVIDIA_VISIBLE_DEVICES=4 \
--mount type=bind,source=${PWD},target=/app \
--entrypoint /bin/bash \
 parcel2d_cnn_gpu:latest'