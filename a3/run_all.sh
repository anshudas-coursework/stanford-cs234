#!/bin/bash
for envs in pendulum cartpole cheetah
do 
    for runs in no-baseline baseline ppo
    do
        for seeds in 1 2 3
        do
            clear
            python code/main.py --env-name $envs --$runs --seed $seeds
        done
    done
done