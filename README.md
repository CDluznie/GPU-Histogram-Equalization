# GPU-Histogram-Equalization

An efficient histogram equalizer on GPU with C++/CUDA.
The parallel implementation allows the program to handle big PNG files.

<p align="center">
![alt-text-1](./img/cars.png) ![alt-text-2](./img/cars_he.png)
</p>

## Requirements

* g++
* nvcc

## Usage

* Compilation : `make`
* Run : `./histogram_equalizer inputImageName outputName`
    * *inputImageName* : image to process, must be a PNG file
    * *outputName* : name of the output without extension
* Delete binaries : `make clean`
