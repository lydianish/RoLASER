#!/bin/bash

set -e

if [ -z ${LASER} ] ; then
  echo "Please set the environment variable 'LASER'"
  exit 1
fi

# Download models
echo Downloading models...

mkdir -p models
cd models

wget https://zenodo.org/api/records/10864557/files-archive
unzip files-archive

# LASER
mkdir -p $LASER/models
mv laser* $LASER/models/

# RoLASER
mkdir RoLASER
mv rolaser* RoLASER/

# c-RoLASER
mkdir c-RoLASER
mv c-rolaser* c-RoLASER/

rm files-archive
cd ..

echo Done...