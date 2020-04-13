#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <chrono>
#include <cassert>
#include <vector>

#include "lodepng.h"

cudaError_t status;
#define CudaCall(x) \
	status = x; \
	if (status != cudaSuccess) \
		std::cout << "Error [" << status << "]: " << cudaGetErrorString(status) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl


// GPU Kernel functions
#pragma region gpuCode

__global__
void ScaleAndGray(unsigned char* orig, unsigned* gray, unsigned width, unsigned height, int scaleFactor) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= height || j >= width)
		return;

	int newWidth = width / scaleFactor;

	int x = (scaleFactor * i - 1 * (i > 0));
	int y = (scaleFactor * j - 1 * (j > 0));

	gray[i * newWidth + j] =
		0.3 * orig[x * (4 * width) + 4 * y] +
		0.59 * orig[x * (4 * width) + 4 * y + 1] +
		0.11 * orig[x * (4 * width) + 4 * y + 2];
}

__global__
void Zncc(unsigned* leftPixels, unsigned* rightPixels, unsigned* disparityMap, unsigned width, unsigned height,
	int minDisp, int maxDisp, int windowWidth, int windowHeight) {
	
	unsigned windowSize = windowWidth * windowHeight;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= height || j >= width)
		return;

	float bestDisparity = maxDisp;
	float bestZncc = -1;

	// Select the best disparity value for the current pixel
	for (int d = minDisp; d <= maxDisp; d++) {
		// Calculating mean of blocks using the sliding window method
		float meanLBlock = 0, meanRBlock = 0;

		for (int x = -windowHeight / 2; x < windowHeight / 2; x++) {
			for (int y = -windowWidth / 2; y < windowWidth / 2; y++) {
				// Check for image borders
				if (
					!(i + x >= 0) ||
					!(i + x < height) ||
					!(j + y >= 0) ||
					!(j + y < width) ||
					!(j + y - d >= 0) ||
					!(j + y - d < width)
					) {
					continue;
				}

				meanLBlock += leftPixels[(i + x) * width + (j + y)];
				meanRBlock += rightPixels[(i + x) * width + (j + y - d)];
			}
		}

		meanLBlock /= windowSize;
		meanRBlock /= windowSize;

		// Calculate ZNCC for current disparity value
		float stdLBlock = 0, stdRBlock = 0;
		float currentZncc = 0;

		for (int x = -windowHeight / 2; x < windowHeight / 2; x++) {
			for (int y = -windowWidth / 2; y < windowWidth / 2; y++) {
				// Check for image borders
				if (
					!(i + x >= 0) ||
					!(i + x < height) ||
					!(j + y >= 0) ||
					!(j + y < width) ||
					!(j + y - d >= 0) ||
					!(j + y - d < width)
					) {
					continue;
				}

				int centerL = leftPixels[(i + x) * width + (j + y)] - meanLBlock;
				int centerR = rightPixels[(i + x) * width + (j + y - d)] - meanRBlock;

				// standard deviation
				stdLBlock += centerL * centerL;
				stdRBlock += centerR * centerR;

				currentZncc += centerL * centerR;
			}
		}

		currentZncc /= sqrtf(stdLBlock) * sqrtf(stdRBlock);

		// Selecting best disparity
		if (currentZncc > bestZncc) {
			bestZncc = currentZncc;
			bestDisparity = d;
		}
	}

	disparityMap[i * width + j] = (unsigned)fabs(bestDisparity);
}

__global__
void CrossCheck(unsigned* leftDisp, unsigned* rightDisp, unsigned* result, unsigned imSize, int crossCheckingThreshold) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= imSize)
		return;

	int diff = leftDisp[i] - rightDisp[i];
	if (diff >= 0) { // leftDisp is winner
		if (diff <= crossCheckingThreshold) {
			result[i] = leftDisp[i];
		} else {
			result[i] = 0;
		}
	} else { //  rightDisp is winner
		if (-diff <= crossCheckingThreshold) {
			result[i] = rightDisp[i];
		} else {
			result[i] = 0;
		}

	}
}

__global__
void OcclusionFill(unsigned* map, unsigned* result, unsigned width, unsigned height, int occlusionNeighbours) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= height || j >= width)
		return;

	unsigned currentIndex = i * width + j;
	result[currentIndex] = map[currentIndex];

	// If the pixel value is 0, copy value from nearest non zero neighbour
	if (map[currentIndex] == 0) {
		bool stop = false;

		for (int n = 1; n <= occlusionNeighbours / 2 && !stop; n++) {
			for (int y = -n; y <= n && !stop; y++) {
				for (int x = -n; x <= n && !stop; x++) {
					// Checking for borders
					if (
						!(i + x >= 0) ||
						!(i + x < height) ||
						!(j + y >= 0) ||
						!(j + y < width) ||
						(x == 0 && y == 0)
						) {
						continue;
					}

					int index = (i + x) * width + (j + y);

					if (map[index] == 0) {
						result[currentIndex] = map[index];
						stop = true;
						break;
					}
				}
			}
		}
	}
}

#pragma endregion gpuCode


/*
Class to calculate time taken by functions in seconds.
* Creating an object of the class in a function, calls the constructor which starts the timer.
* At the end of the function, the destructor is called which stops the timer and calculates the duration.
* We can get the duration manually using the getElapsedTime method.
*/
class Timer {
private:
	std::chrono::time_point<std::chrono::steady_clock> m_Start, m_End;
	std::chrono::duration<float> m_Duration;

public:
	Timer() {
		m_Start = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		m_End = std::chrono::high_resolution_clock::now();
		m_Duration = m_End - m_Start;

		std::cout << "Done (" << m_Duration.count() << " s)" << std::endl;
	}

	float getElapsedTime() {
		m_End = std::chrono::high_resolution_clock::now();
		m_Duration = m_End - m_Start;

		return m_Duration.count();
	}
};

constexpr int maxDisparity = 64;

constexpr int windowWidth = 15;
constexpr int windowHeight = 15;

constexpr int crossCheckingThreshold = 2;

constexpr int occlusionNeighbours = 256;

constexpr int scaleFactor = 4;

std::vector<unsigned char> loadImage(const char*, unsigned&, unsigned&);
std::vector<unsigned char> normalize(std::vector<unsigned>, const unsigned, const unsigned);

// Display GPU info
// https://stackoverflow.com/a/5689133
void DisplayHeader() {
	const int kb = 1024;
	const int mb = kb * kb;
	std::cout << "NBody.GPU" << std::endl << "=========" << std::endl << std::endl;

	std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;

	int devCount;
	cudaGetDeviceCount(&devCount);
	std::cout << "CUDA Devices: " << std::endl << std::endl;

	for (int i = 0; i < devCount; ++i) {
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << std::endl;
		std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
		std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << std::endl;
		std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << std::endl;
		std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

		std::cout << "  Warp size:         " << props.warpSize << std::endl;
		std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
		std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
		std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << std::endl;
		std::cout << std::endl;
	}
}

int main() {
	Timer timer;

	DisplayHeader();

	// Host variables
	std::vector<unsigned char> leftPixels, rightPixels;
	unsigned width, height, rightWidth, rightHeight;

	std::cout << "Reading Left Image...";
	leftPixels = loadImage("imageL.png", width, height);

	std::cout << "Reading Right Image...";
	rightPixels = loadImage("imageR.png", rightWidth, rightHeight);

	// left and right images are assumed to be of same dimensions
	assert(width == rightWidth && height == rightHeight);

	width /= scaleFactor;
	height /= scaleFactor;

	unsigned imSize = width * height;
	unsigned origSize = rightWidth * rightHeight;
	std::vector<unsigned> output(imSize);

	// Device variabels
	unsigned char *d_origL, *d_origR;
	unsigned *d_grayL, *d_grayR, *d_dispLR, *d_dispRL, *d_dispCC, *d_output;

	CudaCall(cudaMalloc((void**) &d_origL, sizeof(unsigned char) * origSize * 4));
	CudaCall(cudaMalloc((void**) &d_origR, sizeof(unsigned char) * origSize * 4));
	CudaCall(cudaMalloc((void**) &d_grayL, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**) &d_grayR, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**) &d_dispLR, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**) &d_dispRL, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**) &d_dispCC, sizeof(unsigned) * imSize));
	CudaCall(cudaMalloc((void**) &d_output, sizeof(unsigned) * imSize));

	// Copy Data from host to device
	CudaCall(cudaMemcpy(d_origL, leftPixels.data(), sizeof(leftPixels[0]) * leftPixels.size(), cudaMemcpyHostToDevice));
	CudaCall(cudaMemcpy(d_origR, rightPixels.data(), sizeof(rightPixels[0]) * rightPixels.size(), cudaMemcpyHostToDevice));

	// Profiling
	float elapsed = 0;
	cudaEvent_t start, stop;

	CudaCall(cudaEventCreate(&start));
	CudaCall(cudaEventCreate(&stop));

	// Kernel Calls
	dim3 blocks(height / 21, width / 21);
	dim3 threads(21, 21);
	dim3 blocks1D((height / 21) * (width / 21));
	dim3 threads1D(21 * 21);

	// Scale and Gray left
	std::cout << "Converting Left Image to grayscale...";
	CudaCall(cudaEventRecord(start));
	
	ScaleAndGray<<<blocks, threads>>>(d_origL, d_grayL, rightWidth, rightHeight, scaleFactor);
	
	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Scale and Gray right
	std::cout << "Converting Right Image to grayscale...";
	CudaCall(cudaEventRecord(start));

	ScaleAndGray<<<blocks, threads>>>(d_origR, d_grayR, rightWidth, rightHeight, scaleFactor);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Disparity Left over Right
	std::cout << "Converting Left Disparity Map...";
	CudaCall(cudaEventRecord(start));

	Zncc<<<blocks, threads>>>(d_grayL, d_grayR, d_dispLR, width, height, 0, maxDisparity, windowWidth, windowHeight);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Disparity Right over Left
	std::cout << "Converting Right Disparity Map...";
	CudaCall(cudaEventRecord(start));

	Zncc<<<blocks, threads>>>(d_grayR, d_grayL, d_dispRL, width, height, -maxDisparity, 0, windowWidth, windowHeight);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Cross Checking
	std::cout << "Performing Cross Checking...";
	CudaCall(cudaEventRecord(start));

	CrossCheck<<<blocks1D, threads1D>>>(d_dispLR, d_dispRL, d_dispCC, imSize, crossCheckingThreshold);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Occlusion Filling
	std::cout << "Performing Occlusion Filling...";
	CudaCall(cudaEventRecord(start));

	OcclusionFill<<<blocks, threads>>>(d_dispCC, d_output, width, height, occlusionNeighbours);

	CudaCall(cudaEventRecord(stop));
	CudaCall(cudaEventSynchronize(stop));
	CudaCall(cudaEventElapsedTime(&elapsed, start, stop));
	std::cout << "Done (" << elapsed / 1000 << " s)" << std::endl;

	CudaCall(cudaPeekAtLastError());
	CudaCall(cudaDeviceSynchronize());

	// Copy data from device to host
	CudaCall(cudaMemcpy(&output[0], d_output, sizeof(unsigned) * imSize, cudaMemcpyDeviceToHost));

	lodepng::encode("output.png", normalize(output, width, height), width, height);

	std::cout << "The program took " << timer.getElapsedTime() << " s" << std::endl;

	cudaFree(d_origL);
	cudaFree(d_origR);
	cudaFree(d_grayL);
	cudaFree(d_grayR);
	cudaFree(d_dispLR);
	cudaFree(d_dispRL);
	cudaFree(d_dispCC);
	cudaFree(d_output);

	std::cin.get();
	return 0;
}

std::vector<unsigned char> loadImage(const char* filename, unsigned& width, unsigned& height) {
	Timer timer;

	std::vector<unsigned char> pixels;

	unsigned error = lodepng::decode(pixels, width, height, filename);
	if (error) {
		std::cout << "Failed to load image: " << lodepng_error_text(error) << std::endl;
		std::cin.get();
		exit(-1);
	}

	return pixels;
}

std::vector<unsigned char> normalize(
	std::vector<unsigned> in,
	const unsigned width,
	const unsigned height
) {
	std::vector<unsigned char> result(width * height * 4);

	unsigned char max = 0;
	unsigned char min = UCHAR_MAX;

	for (int i = 0; i < width * height; i++) {
		if (in[i] > max) {
			max = in[i];
		}

		if (in[i] < min) {
			min = in[i];
		}
	}

	// Normalize values to be between 0 and 255
	int mapIndex = 0;
	for (int i = 0; i < width * height * 4; i += 4, mapIndex++) {
		result[i] = result[i + 1] = result[i + 2] = (unsigned char)(255 * (in[mapIndex] - min) / (max - min));
		result[i + 3] = 255;
	}

	return result;
}
