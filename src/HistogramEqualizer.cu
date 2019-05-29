#include "HistogramEqualizer.hpp"

namespace HistogramEqualizationGPU {
	
	// The number of level in the input image value channel histogram
	static const int HISTOGRAM_LEVELS = 256;
	
	// The number of levels managed by one thread in the histogram kernel computation
	static const int BIN_PER_THREAD = 4;
	
	// The 2D texture for the input image
	texture<uchar4,cudaTextureType2D> tex2d_image;
	
	// The 1D texture for the cumulated histogram
	texture<float> tex_cumulated_histogram;	
	
	// Compute the value of a float x modulo y
	// x : input float value
	// y : input float value
	__device__
	float floating_modulo(const float x, const float y) {
		float result = x;
		while (result >= 0.f) {
			result -= y;
		}
		return (result+y);
	}
	
	// Compute the index in the image array of a pixel
	// x : abscissa of the pixel
	// y : ordinate of the pixel
	// imgWitdh : width of the image
	// imgHeight : height of the image
	__device__
	int computeImageIndex(const int x, const int y, const uint imgWidth, const uint imgHeight) {
		return y * imgWidth + x;
	}
	
	// Compute the index in the histogram of value channel color
	// value : value channel color
	__device__
	int computeBinIndex(const float value) {
		return int(round(value*(HISTOGRAM_LEVELS-1.f)));
	}
	
	// Cuda kernel to compute the HSV transform of an RGB image
	// The input image is read from the texture 'tex2d_image'
	// The differents channels are stored into differents arrays
	// The alpha channel is unmodified
	// imgWidth : width of the image
	// imgHeight : height of the image
	// imgHue : allocated array for hue channel, filled by the kernel
	// imgSaturation : allocated array for saturation channel, filled by the kernel
	// imgValue : allocated array for value channel, filled by the kernel
	// imgAlpha : allocated array for alpha channel, filled by the kernel
	__global__ 
	void rgbToHsv(const uint imgWidth, const uint imgHeight, float * imgHue, float * imgSaturation, float * imgValue, uchar * imgAlpha) {
		
		for (int y = blockIdx.y*blockDim.y + threadIdx.y; y < imgHeight; y += blockDim.y*gridDim.y) {
			
			for (int x = blockIdx.x*blockDim.x + threadIdx.x; x < imgWidth; x += blockDim.x*gridDim.x) {
		
				const int index_image = computeImageIndex(x, y, imgWidth, imgHeight);
				const uchar4 pixel_image = tex2D(tex2d_image, x, y); // Read pixel (x,y) in 2D texture
				
				// Compute HSV channels
				const float rr = pixel_image.x/255.f;
				const float gg = pixel_image.y/255.f;
				const float bb = pixel_image.z/255.f;
				const float cmax = fmax(max(rr,gg),bb);
				const float cmin = fmin(min(rr,gg),bb);
				const float delta = cmax - cmin;
				float hue, saturation, value;
				if (delta == 0.f) {
					hue = 0.f;
				} else if (cmax == rr) {
					hue = 60.f*floating_modulo((gg-bb)/delta, 6.f);
				} else if (cmax == gg) {
					hue = 60.f*(((bb-rr)/delta) + 2.f);
				} else {
					hue = 60.f*(((rr-gg)/delta) + 4.f);
				}
				if (cmax == 0.f) {
					saturation = 0.f;
				} else {
					saturation = delta/cmax;
				}
				value = cmax;
				
				// Store channels in the dedicated arrays
				imgHue[index_image] = hue;
				imgSaturation[index_image] = saturation;
				imgValue[index_image] = value;
				imgAlpha[index_image] = pixel_image.w; // Unmodified alpha channel
				
			}	
			
		}
		
	}
	
	// Cuda kernel to compute the value channel histogram of the image
	// The histogram is normalized
	// The kernel should be call with 'HISTOGRAM_LEVELS' / 'BIN_PER_THREAD' threads
	// imgValue : value channel of the image
	// imgWidth : width of the image
	// imgHeight : height of the image
	// histogram : allocated array for histogram, filled by the kernel
	__global__
	void computeHistogram(const float * imgValue, const uint imgWidth, const uint imgHeight, float * histogram) {
		
		// Use shared memory to store block-local histogram
		__shared__ float shared_local_histogram[HISTOGRAM_LEVELS];
		
		// Compute an incrementation to make the histogram normalized at the end of the computation
		// and avoid a post-normalization
		const float incrementation = (HISTOGRAM_LEVELS - 1.f)/(HISTOGRAM_LEVELS*imgWidth*imgHeight);
		
		// Initialize all the histogram levels managed by the thread to 0
		#pragma unroll
		for (int offset = 0; offset < HISTOGRAM_LEVELS; offset += (HISTOGRAM_LEVELS/BIN_PER_THREAD)) {
			shared_local_histogram[threadIdx.x + offset] = 0;
		}
		__syncthreads();
		
		// Add in the local histogram the pixels of the blocks 
		for (int index_image = blockIdx.x*blockDim.x + threadIdx.x; index_image < imgWidth*imgHeight; index_image += blockDim.x*gridDim.x) {
			const int histogram_index = computeBinIndex(imgValue[index_image]);
			// Use atomic addition because same threads in the block can increment the same level
			atomicAdd(shared_local_histogram + histogram_index, incrementation);
		}
		__syncthreads();
		
		// Merge all the block-local histogram to global histogram (using atomic add)
		// Add the local value of a level to the global value of a level, for all levels managed by the thread
		#pragma unroll
		for (int offset = 0; offset < HISTOGRAM_LEVELS; offset += (HISTOGRAM_LEVELS/BIN_PER_THREAD)) {
			if (shared_local_histogram[threadIdx.x + offset] != 0) { // To avoid useless atomic add
				// atomic add because same level of differents block can be increment at same time
				atomicAdd(histogram + threadIdx.x + offset, shared_local_histogram[threadIdx.x + offset]);
			}
		}
	
	}
	
	// Cuda kernel to compute the cumulated histogram of value channel
	// The kernel should be call with 'HISTOGRAM_LEVELS' threads and 1 block
	// histogram : histogram of value channel
	// cumulatedHistogram : allocated array for cumulated histogram, filled by the kernel
	__global__
	void computeCumulatedHistogram(float * histogram, float * cumulatedHistogram) {
		
		// Use shared memory to make the cumulated histogram computation
		__shared__ float shared_cumulated_histogram[HISTOGRAM_LEVELS];
		
		// Copy into the shared memory the histogram
		shared_cumulated_histogram[threadIdx.x] = histogram[threadIdx.x];
		__syncthreads(); 
		
		// Use hillis steele scan to compute cumulated histogram
		for (int offset = 1; offset < HISTOGRAM_LEVELS; offset *= 2) { 
			if (threadIdx.x >= offset) {
				shared_cumulated_histogram[threadIdx.x] += shared_cumulated_histogram[threadIdx.x - offset]; 	
			}
			__syncthreads(); 
		}
		
		// Copy the result
		cumulatedHistogram[threadIdx.x] = shared_cumulated_histogram[threadIdx.x];
	
	}
	
	// Cuda kernel to compute the histogram equalization on value channel
	// The transformation is make in place
	// The cumulated histogram is read from 1D texture 'tex_cumulated_histogram'
	// imgValue : value channel of the image, updated by the kernel
	// imgWidth : width of the image
	// imgHeight : height of the image
	__global__
	void computeImageEqualization(float * imgValue, const uint imgWidth, const uint imgHeight, const float * cumulatedHistogram) {
		
		for (int y = blockIdx.y*blockDim.y + threadIdx.y; y < imgHeight; y += blockDim.y*gridDim.y) {
		
			for (int x = blockIdx.x*blockDim.x + threadIdx.x; x < imgWidth; x += blockDim.x*gridDim.x) {
		
				const int index_image = computeImageIndex(x, y, imgWidth, imgHeight);
				const int histogram_index = computeBinIndex(imgValue[index_image]);
				
				// The equalization change a channel value by the number of his level in the cumulated histogram  
				//imgValue[index_image] = tex1Dfetch(tex_cumulated_histogram, histogram_index); // Read histogram_index from tex_cumulated_histogram
				imgValue[index_image] = cumulatedHistogram[histogram_index]; // Should read in texture if it works
		
			} 
		
		}
	
	}
	
	// Cuda kernel to compute the RGB transform of an HSV image
	// The alpha channel is unmodified
	// imgHue : hue channel of the image
	// imgSaturation : saturation channel of the image
	// imgValue : value channel of the image
 	// imgAlpha : alpha channel of the image
	// imgWidth : width of the image
	// imgHeight : height of the image
	// img : allocated array for RGB image, filled by the kernel
	__global__ 
	void hsvToRgb(const float * imgHue, const float * imgSaturation, const float * imgValue, const uchar * imgAlpha, const uint imgWidth, const uint imgHeight, uchar4 * img) {					
		
		for (int y = blockIdx.y*blockDim.y + threadIdx.y; y < imgHeight; y += blockDim.y*gridDim.y) {
		
			for (int x = blockIdx.x*blockDim.x + threadIdx.x; x < imgWidth; x += blockDim.x*gridDim.x) {
		
				const int index_image = computeImageIndex(x, y, imgWidth, imgHeight);
				
				// Compute RGB transform
				const float c = imgValue[index_image]*imgSaturation[index_image];
				const float hh = imgHue[index_image]/60.f;
				const float xx = c * (1.f - abs(floating_modulo(hh, 2.f) - 1.f));
				float rr, gg, bb;
				if (0.f <= hh && hh < 1.f) {
					rr = c;
					gg = xx;
					bb = 0.f;
				} else if (1.f <= hh && hh < 2.f) {
					rr = xx;
					gg = c;
					bb = 0.f;
				} else if (2.f <= hh && hh < 3.f) {
					rr = 0.f;
					gg = c;
					bb = xx;
				} else if (3.f <= hh && hh < 4.f) {
					rr = 0.f;
					gg = xx;
					bb = c;
				} else if (4.f <= hh && hh < 5.f) {
					rr = xx;
					gg = 0.f;
					bb = c;
				} else if (5.f <= hh && hh < 6.f) {
					rr = c;
					gg = 0.f;
					bb = xx;
				} else {
					rr = 0.f;
					gg = 0.f;
					bb = 0.f;
				}
				const float m = imgValue[index_image] - c;
				const uchar r = uchar(255*(rr + m));
				const uchar g = uchar(255*(gg + m));
				const uchar b = uchar(255*(bb + m));
				
				// Store the final result
				img[index_image].x = r;
				img[index_image].y = g;
				img[index_image].z = b;
				img[index_image].w = imgAlpha[index_image]; // Unmodified alpha channel
			
			}	
		
		}
	
	}

	// Compute the histogram equalization on GPU of a RGB image
	// inputImg : input image
	// imgWidth : width of the image
	// imgHeight : height of the image
	// outputImg : allocated vector for the result, filled by the function
	void HistogramEqualization(const std::vector<uchar4> &inputImg, const uint imgWidth, const uint imgHeight, std::vector<uchar4> &outputImg) {
		
		uchar4 *dev_input_image = NULL; // // GPU input image (for 2D texture)
		float *dev_hue_image = NULL; // GPU hue channel of image
		float *dev_saturation_image = NULL; // GPU saturation channel of image
		float *dev_value_image = NULL; // GPU value channel of image
		uchar *dev_alpha_image = NULL; // GPU alpha channel of image
		float *dev_histogram = NULL; //GPU histogram image
		float *dev_cumulated_histogram = NULL; //GPU histogram image
		uchar4 *dev_output_image = NULL; //GPU output image
		
		// Channel descriptor for texture binding
		cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<uchar4>();
		
		const size_t image_bytes = imgWidth * imgHeight * sizeof(uchar4); 
		const size_t channel_hsv_bytes = imgWidth * imgHeight * sizeof(float); 
		const size_t channel_alpha_bytes = imgWidth * imgHeight * sizeof(uchar); 
		const size_t histogram_bytes = HISTOGRAM_LEVELS * sizeof(float); 
		const size_t image_width_bytes = imgWidth * sizeof(uchar4); // Width for 2D image allocation
		
		size_t pitch; // Pitch for 2D texture
		
		// Number of threads and blocks for 2D processing kernels
		const int nb_thread_x = 32;
		const int nb_thread_y = 32;
		const int nb_block_x = (imgWidth - 1)/nb_thread_x + 1;
		const int nb_block_y = (imgHeight - 1)/nb_thread_y + 1;
		
		// Number of threads and blocks for computeHistogram kernel
		const int nb_thread = (HISTOGRAM_LEVELS/BIN_PER_THREAD);
		const int nb_block = (imgWidth*imgHeight - 1)/nb_thread + 1;
		
		// Dynamic GPU array allocation
		HandleError(cudaMallocPitch(&dev_input_image, &pitch, image_width_bytes, imgHeight));
		HandleError(cudaMalloc(&dev_hue_image, channel_hsv_bytes));
		HandleError(cudaMalloc(&dev_saturation_image, channel_hsv_bytes));
		HandleError(cudaMalloc(&dev_value_image, channel_hsv_bytes));
		HandleError(cudaMalloc(&dev_alpha_image, channel_alpha_bytes));
		HandleError(cudaMalloc(&dev_output_image, image_bytes));
		HandleError(cudaMalloc(&dev_histogram, histogram_bytes));
		HandleError(cudaMalloc(&dev_cumulated_histogram, histogram_bytes));
		
		// Dynamic GPU array initialization
		HandleError(cudaMemcpy2D(dev_input_image, pitch, inputImg.data(), image_width_bytes, image_width_bytes, imgHeight, cudaMemcpyHostToDevice));
		HandleError(cudaMemset(dev_histogram, 0, histogram_bytes));

		// Bind the input image to 2D texture
		HandleError(cudaBindTexture2D(0, tex2d_image, dev_input_image, channel_desc, imgWidth, imgHeight, pitch));
		
		// Convert image to HSV space
		rgbToHsv<<<dim3(nb_block_x, nb_block_y), dim3(nb_thread_x, nb_thread_y)>>>(
			imgWidth, imgHeight,
			dev_hue_image, dev_saturation_image, dev_value_image, dev_alpha_image
		);
		
		// Compute histogram on value channel
		computeHistogram<<<dim3(nb_block), dim3(nb_thread)>>>(
			dev_value_image,
			imgWidth, imgHeight,
			dev_histogram
		);

		// Compute cumulated histogram
		computeCumulatedHistogram<<<dim3(1), dim3(HISTOGRAM_LEVELS)>>>(
			dev_histogram,
			dev_cumulated_histogram
		);

		// Bind the cumulated histogram to 1D texture 
		HandleError(cudaBindTexture(0, tex_cumulated_histogram, dev_cumulated_histogram, histogram_bytes));
		
		// Compute histogram equalization on value channel
		computeImageEqualization<<<dim3(nb_block_x, nb_block_y), dim3(nb_thread_x, nb_thread_y)>>>(
			dev_value_image,
			imgWidth, imgHeight,
			dev_cumulated_histogram
		);

		// Compute back image to RGB space
		hsvToRgb<<<dim3(nb_block_x, nb_block_y), dim3(nb_thread_x, nb_thread_y)>>>(
			dev_hue_image, dev_saturation_image, dev_value_image, dev_alpha_image,
			imgWidth, imgHeight,
			dev_output_image
		);
		
		// Copy result
		HandleError(cudaMemcpy(outputImg.data(), dev_output_image, image_bytes, cudaMemcpyDeviceToHost));

		// Free memory
		HandleError(cudaFree(dev_input_image));
		HandleError(cudaFree(dev_output_image));
		HandleError(cudaFree(dev_hue_image));
		HandleError(cudaFree(dev_saturation_image));
		HandleError(cudaFree(dev_value_image));
		HandleError(cudaFree(dev_alpha_image));
		HandleError(cudaFree(dev_histogram));
		HandleError(cudaFree(dev_cumulated_histogram));
		
		// Unbind textures
		HandleError(cudaUnbindTexture(tex2d_image));
		HandleError(cudaUnbindTexture(tex_cumulated_histogram));
		
	}
	
}
