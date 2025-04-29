#!/bin/bash

container=$1
world_port=$2
tra_port=$3
radar=($4)
interval=$5
vehicle=$6
walker=$7
rawdata_path=$8
res=$9

if [ -n "$(docker ps -a --filter "name=$container" --quiet)" ]; then
	echo "Carla Server 啟動中..."
	docker start $container
	sleep 15
else
	echo "First time Carla Server 啟動中..."
	docker run -d --name $container --restart unless-stopped --privileged --gpus all --net=host -e DISPLAY=:8 -v /tmp/.X11-unix:/tmp/.X11-unix:rw carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -vulkan -RenderOffScreen -world-port=$world_port
	sleep 30
fi

dockerstop(){
	echo "Carla Server 停止中..."
	docker stop $container
	sleep 3
	screen -X quit
	exit 0
}

trap dockerstop SIGINT

cd ..

while true; do
	python generate_raw_data.py --sync --radar ${radar[0]} ${radar[1]} ${radar[2]} ${radar[3]} ${radar[4]} --interval $interval -n $vehicle -w $walker --res $res --port $world_port --traport $tra_port --rawdata-path $rawdata_path --autopilot --projectionflag
	if [ $? -eq 134 ]; then
		echo "Python script aborted (core dumped). Restarting..."
	fi
	sleep 1
done

