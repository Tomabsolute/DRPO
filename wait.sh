#!/bin/bash

while true
do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)

    if [ $FREE -gt 20000 ]; then
        echo "GPU free, starting job"
        break
    else
        echo "GPU busy, waiting..."
        sleep 60
    fi
done

accelerate launch --main_process_port 0 -m examples.tldr.drpo

    sleep 60
done