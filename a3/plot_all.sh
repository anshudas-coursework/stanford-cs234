#!/bin/bash
for envs in pendulum cartpole cheetah
do
    clear
    python code/plot.py --env-name $envs --seeds 1,2,3
done