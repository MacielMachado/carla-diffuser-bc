
#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/
export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
# cat > logfile.txt
# screen -L logfile.txt -S eval_carla_bc .venv/bin/python eval_diffusion_bc.py
screen -L -S eval_carla_bc .venv/bin/python eval_diffusion_bc_fixed.py
