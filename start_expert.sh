#!/bin/bash

# export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/
# export PYTHONPATH=$PYTHONPATH:./api_carla/9.10/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:/home/joey/workspace/carla_0914/PythonAPI/carla
# export PYTHONPATH=$PYTHONPATH:/home/joey/workspace/carla_0914/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg

screen -L -S carla_expert .venv/bin/python data_collect.py