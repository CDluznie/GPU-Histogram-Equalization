#ifndef HISTOGRAM_EQUALIZER_HPP
#define HISTOGRAM_EQUALIZER_HPP

#include <vector>
#include "common.hpp"

namespace HistogramEqualizationGPU {
	
	void HistogramEqualization(const std::vector<uchar4> &inputImg, const uint imgWidth, const uint imgHeight, std::vector<uchar4> &outputImg);

}

#endif
