#include "common.hpp"

std::ostream & operator<<(std::ostream & os, const uchar4 & uc) {  
	return (os << "(" << int(uc.x) << "," << int(uc.y) << "," << int(uc.z) << "," << int(uc.w) <<  ")");  
} 

static void HandleError(cudaError_t err, const char *file, const int line) {
    if (err != cudaSuccess) {
    	std::stringstream ss;
    	ss << line;
        std::string errMsg(cudaGetErrorString(err));
        std::cout 	<< errMsg 
					<< " (file: " + std::string(file)
					<< " at line: " + ss.str() + ")"
					<< std::endl;
        throw std::runtime_error(errMsg);
    }
}

void HandleError(cudaError_t err) {
    HandleError(err, __FILE__, __LINE__);
}
