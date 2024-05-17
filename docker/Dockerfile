FROM ubuntu:jammy
ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y locales lsb-release
ARG DEBIAN_FRONTEND=noninteractive
RUN dpkg-reconfigure locales


ENV ROS_DISTRO humble
ARG INSTALL_PACKAGE=desktop
RUN apt-get update -q && \
    apt-get install -y curl gnupg2 lsb-release && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update -q && \
    apt-get install -y ros-${ROS_DISTRO}-${INSTALL_PACKAGE} \
    python3-argcomplete \
    python3-colcon-common-extensions \
    python3-rosdep python3-vcstool && \
    rosdep init && \
    rm -rf /var/lib/apt/lists/*


RUN mkdir -p /ros2_ws/src 
# RUN cd /ros2_ws; ./opt/ros/humble/setup.bash
# RUN cd /ros2_ws; colcon build --symlink-install --executor sequential

RUN rosdep update
ARG UNAME=hariharan
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} -o${UNAME}
RUN useradd -m -u ${UID} -o -s /bin/bash ${UNAME}
RUN echo 'hariharan:hariharan' | chpasswd
RUN adduser ${UNAME} sudo
WORKDIR /home/hariharan/ros2_ws
RUN cp -r /ros2_ws /home/hariharan
RUN chown -R $UNAME:${UNAME} /home/hariharan
RUN chmod 775 /home/hariharan
RUN echo "source /opt/ros/humble/setup.bash" >> /home/hariharan/.bashrc
RUN echo "source /home/hariharan/ros2_ws/install/setup.bash" >> /home/hariharan/.bashrc
RUN apt-get update; apt-get install tmux -y
RUN apt-get install nano -y
#RUN source /opt/ros/humble/setup.bash
RUN apt install ros-humble-xacro -y
RUN apt install ros-humble-ros2-control -y
RUN apt install ros-humble-ros2-controllers -y
RUN apt install ros-humble-gazebo-ros2-control -y
RUN apt-get update 
RUN apt-get install build-essential
RUN cd home/hariharan/ros2_ws;sudo chmod 777 -R .;colcon build --symlink-install --executor sequential
USER root 
ENTRYPOINT [ "/bin/bash" , "/home/hariharan/ros2_ws/src/docker/entrypoint.sh" ]
