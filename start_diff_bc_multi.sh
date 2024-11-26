#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg

if [ "$1" == "-new_arch" ]; then
    script_name="learn_diffusion_bc_multimodality_birdview_new_arch.py"
    screen_name="carla_bc_new_arch"
else
    script_name="learn_diffusion_bc_multimodality_birdview.py"
    screen_name="carla_bc"
fi

screen -L -S "$screen_name" .venv/bin/python "$script_name"