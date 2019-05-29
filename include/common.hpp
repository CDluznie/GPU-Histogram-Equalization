#ifndef __COMMON_HPP
#define __COMMON_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

typedef unsigned char uchar;
typedef unsigned int uint;

void HandleError(cudaError_t err);

std::ostream & operator<<(std::ostream & os, const uchar4 & uc);

#endif



