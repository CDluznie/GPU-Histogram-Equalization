#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <iomanip>     
#include <cstring>
#include <exception>
#include "lodepng.h"
#include "HistogramEqualizer.hpp"

namespace HistogramEqualizationGPU {
	
	void loadImage(const std::string & fname, std::vector<uchar4> & img, uint & imgWidth, uint & imgHeight) {
		std::vector<uchar> imgUchar;
		unsigned error = lodepng::decode(imgUchar, imgWidth, imgHeight, fname, LCT_RGBA);
		if (error) {
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
		img.resize(imgUchar.size() / 4);
		for (uint i = 0; i < img.size(); ++i) {
			const uint id = 4 * i;
			img[i].x = imgUchar[id];
			img[i].y = imgUchar[id + 1];
			img[i].z = imgUchar[id + 2];
			img[i].w = imgUchar[id + 3];
		}
	}

	void saveImage(const std::string & fname, const std::vector<uchar4> & img, const uint & imgWidth, const uint & imgHeight) {
		unsigned error = lodepng::encode(fname, reinterpret_cast<const uchar *>(img.data()), imgWidth, imgHeight, LCT_RGBA);
		if (error) {
			throw std::runtime_error("Error loadpng::decode: " + std::string(lodepng_error_text(error)));
		}
	}
		
	void printUsageAndExit(const char *prg)  {
		std::cerr	<< "Usage: " 
					<< prg
					<< " inputImageName"
					<< " outputName"
					<< std::endl;
		exit(EXIT_FAILURE);
	}

	void main(int argc, char **argv) {	
		
		if (argc != 3) {
			std::cerr << "Wrong number of argument" << std::endl;
			printUsageAndExit(argv[0]);
		}
		
		const std::string inputFileName = argv[1];
		const std::string outputFileName = std::string(argv[2]) + ".png";

		uint imgWidth;
		uint imgHeight;

		std::vector<uchar4> inputImg;

		std::cout << "Loading " << inputFileName << std::endl;
		loadImage(inputFileName, inputImg, imgWidth, imgHeight);
		std::cout << "Image has " << imgWidth << " x " << imgHeight << " pixels" << std::endl;

		std::vector<uchar4> outputImg(imgWidth * imgHeight);

		std::cout << "Histogram equalization in progress..." << std::endl;
		HistogramEqualization(inputImg, imgWidth, imgHeight, outputImg);
		std::cout << "Histogram equalization done" << std::endl;

		std::cout << "Save image as " << outputFileName << std::endl;
		saveImage(outputFileName, outputImg, imgWidth, imgHeight);
		
	}
}

int main(int argc, char **argv) {
	
	try {
		HistogramEqualizationGPU::main(argc, argv);
	} catch (const std::exception &e) {
		std::cerr << e.what() << std::endl;
	}
	
}
