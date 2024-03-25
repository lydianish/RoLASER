#!/bin/bash

set -xe

# Download models
echo Downloading models...

mkdir -p models
wget wget https://zenodo.org/api/records/10864557/files-archive -P models

cd models
unzip files-archive

# LASER
mkdir /LASER
mv laser* LASER
cp LASER/* $LASER/models
# RoLASER
mkdir RoLASER
mv rolaser* RoLASER
# c-RoLASER
mkdir /c-RoLASER
mv c-rolaser* c-RoLASER

rm files-archive
cd ..

echo Done...