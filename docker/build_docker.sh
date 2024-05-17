#!/usr/bin/env bash 
image_name=ddom_docker
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t ${image_name} $(dirname "$0")/
