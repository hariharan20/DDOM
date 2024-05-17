image_name=ddom_docker
xhost + local:docker

echo "Starting docker container..."
docker run --privileged --network host \
           --gpus all \
           --env="NVIDIA_DRIVER_CAPABILITIES=all" \
           --env="DISPLAY=$DISPLAY" \
           --env="QT_X11_NO_MITSHM=1" \
           --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
           -v $(pwd)/../:/home/hariharan/ros2_ws/src \
           -v /dev/dri:/dev/dri \
  	   -v /dev/bus/usb:/dev/bus/usb \
	   -v /dev/input:/dev/input \
           --rm \
           -it ${image_name}
