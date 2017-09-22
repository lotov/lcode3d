FROM debian:stretch
MAINTAINER LCODE team "docker@lcode.info"
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
	g++ \
	gcc \
	git \
	libopenmpi-dev \
	make \
	openmpi-bin \
	python3-pip \
	python3-wheel \
	python3.5 \
	python3.5-dev \
	&& apt-get clean
RUN pip3 install 'pip>=8'
RUN python3 -m pip install 'setuptools>=20'
RUN python3 -m pip install 'numpy>=1.8.2'

# Cache packages so that the installation tests reuse cached wheels
RUN python3 -m pip download -d /tmp \
	'cython>=0.21' \
	'flake8>=3.0' \
	'h5py>=2.2' \
	'hacks>=0.0.11' \
	'mako>=1.0' \
	'matplotlib>=1.4' \
	'mpi4py>=1.3' \
	'nose>=1.0' \
	'numpy>=1.8.2' \
	'pylint>=1.5' \
	'scipy>=0.14' \
	&& rm /tmp/*.whl
