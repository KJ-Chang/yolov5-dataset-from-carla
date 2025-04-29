#!/bin/bash
screen_name=${1:-"carla-server-1"}
container=${2:-"carla-server-1"}
world_port=${3:-3000}
tra_port=${4:-8050}
radar="90.0 50000 80 0 25"

interval=${5:-60}
vehicle=${6:-10}
walker=${7:-10}
rawdata_path=${8:-"raw_data"}
res=${9:-"640x360"}

screen -dmS $screen_name

screen -S $screen_name -p 0 -X stuff "conda activate carla$(printf \\r)"
screen -S $screen_name -p 0 -X stuff "bash generate.sh $container $world_port $tra_port '$radar' $interval $vehicle $walker $rawdata_path $res$(printf \\r)"


