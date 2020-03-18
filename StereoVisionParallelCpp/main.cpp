#include <iostream>
#include <cassert>
#include <chrono>
#include <omp.h>

#include "lodepng.h"

/*
Class to calculate time taken by functions in seconds.
* Creating an object of the class in a function, calls the constructor which starts the timer.
* At the end of the function, the destructor is called which stops the timer and calculates the duration.
* We can get the duration manually using the getElapsedTime method.
*/
class Timer {
private:
	std::chrono::time_point<std::chrono::steady_clock> start, end;
	std::chrono::duration<float> duration;

public:
	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;

		std::cout << "Done (" << duration.count() << " s)" << std::endl;
	}

	float getElapsedTime() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;

		return duration.count();
	}
};

constexpr int maxDisparity = 64;

constexpr int windowWidth = 9;
constexpr int windowHeight = 9;

constexpr int crossCheckingThreshold = 2;

constexpr int occlusionNeighbours = 256;

constexpr int scaleFactor = 4;

constexpr int numThreads = 2;

// Prototypes
std::vector<unsigned char> loadImage(const char*, unsigned&, unsigned&);
std::vector<unsigned> scaleAndGray(std::vector<unsigned char>, const unsigned, const unsigned);
std::vector<unsigned> zncc(
	std::vector<unsigned>,
	std::vector<unsigned>,
	const unsigned,
	const unsigned,
	const int,
	const int
);
std::vector<unsigned> crossChecking(
	std::vector<unsigned>,
	std::vector<unsigned>,
	const unsigned,
	const unsigned
);
std::vector<unsigned> occlusionFilling(std::vector<unsigned>, const unsigned, const unsigned);
std::vector<unsigned char> normalize(std::vector<unsigned>, const unsigned, const unsigned);


int main() {
	Timer timer; // For calculating time of entire program

	omp_set_num_threads(numThreads);

	std::vector<unsigned char> leftPixels, rightPixels;
	unsigned width, height, rightWidth, rightHeight;

	std::cout << "Reading Left Image...";
	leftPixels = loadImage("imageL.png", width, height);

	std::cout << "Reading Right Image...";
	rightPixels = loadImage("imageR.png", rightWidth, rightHeight);

	// left and right images are assumed to be of same dimensions
	assert(width == rightWidth && height == rightHeight);

	std::vector<unsigned> grayL = scaleAndGray(leftPixels, width, height);
	std::vector<unsigned> grayR = scaleAndGray(rightPixels, width, height);

	width /= scaleFactor;
	height /= scaleFactor;

	unsigned error = lodepng::encode("grayL.png", normalize(grayL, width, height), width, height);
	error = lodepng::encode("grayR.png", normalize(grayR, width, height), width, height);

	if (error) {
		std::cout << lodepng_error_text(error);
		std::cin.get();
		return -1;
	}

	// Calculate the disparity maps of left over right and vice versa
	std::cout << "Calculating Left Disparity Map...";
	std::vector<unsigned> dispLR = zncc(grayL, grayR, width, height, 0, maxDisparity);

	error = lodepng::encode("dispLR.png", normalize(dispLR, width, height), width, height);

	std::cout << "Calculating Right Disparity Map...";
	std::vector<unsigned> dispRL = zncc(grayR, grayL, width, height, -maxDisparity, 0);

	error = lodepng::encode("dispRL.png", normalize(dispRL, width, height), width, height);

	std::cout << "Performing cross checking...";
	std::vector<unsigned> dispCC = crossChecking(dispLR, dispRL, width, height);

	error = lodepng::encode("dispCC.png", normalize(dispCC, width, height), width, height);

	std::cout << "Performing Occlusion Filling...";
	std::vector<unsigned> ocfill = occlusionFilling(dispCC, width, height);

	error = lodepng::encode("output.png", normalize(ocfill, width, height), width, height);

	std::cout << "The program took " << timer.getElapsedTime() << " s" << std::endl;

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

std::vector<unsigned> scaleAndGray(
	std::vector<unsigned char> origPixels,
	const unsigned width,
	const unsigned height
) {
	unsigned newWidth = width / scaleFactor;
	unsigned newHeight = height / scaleFactor;

	std::vector<unsigned> result(newWidth * newHeight);

	// Downscaling and conversion to grayscale
	#pragma omp parallel for
	for (int i = 0; i < newHeight; i++) {
		for (int j = 0; j < newWidth; j++) {
			int x = (scaleFactor * i - 1 * (i > 0));
			int y = (scaleFactor * j - 1 * (j > 0));

			result[i * newWidth + j] =
				0.3 * origPixels[x * (4 * width) + 4 * y] +
				0.59 * origPixels[x * (4 * width) + 4 * y + 1] +
				0.11 * origPixels[x * (4 * width) + 4 * y + 2];
		}
	}

	return result;
}

std::vector<unsigned> zncc(
	std::vector<unsigned> leftPixels,
	std::vector<unsigned> rightPixels,
	const unsigned width,
	const unsigned height,
	const int minDisp,
	const int maxDisp
) {
	Timer timer;

	std::vector<unsigned> disparityMap(width * height);

	const unsigned windowSize = windowWidth * windowHeight;

	#pragma omp parallel for
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
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

				currentZncc /= sqrt(stdLBlock) * sqrt(stdRBlock);

				// Selecting best disparity
				if (currentZncc > bestZncc) {
					bestZncc = currentZncc;
					bestDisparity = d;
				}
			}

			disparityMap[i * width + j] = (unsigned)abs(bestDisparity);
		}
	}

	return disparityMap;
}

std::vector<unsigned> crossChecking(
	std::vector<unsigned> leftDisp,
	std::vector<unsigned> rightDisp,
	const unsigned width,
	const unsigned height
) {
	Timer timer;

	const unsigned imageSize = width * height;

	std::vector<unsigned> result(imageSize);

	#pragma omp parallel for
	for (int i = 0; i < imageSize; i++) {
		if (abs((int)leftDisp[i] - (int)rightDisp[i]) > crossCheckingThreshold) {
			result[i] = 0;
		} else {
			result[i] = leftDisp[i];
		}
	}

	return result;
}

std::vector<unsigned> occlusionFilling(
	std::vector<unsigned> map,
	const unsigned width,
	const unsigned height
) {
	Timer timer;

	std::vector<unsigned> result(width * height);

	#pragma omp parallel for
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
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

							int currentIndex = (i + x) * width + (j + y);

							if (map[currentIndex] == 0) {
								result[i * width + j] = map[currentIndex];
								stop = true;
								break;
							}
						}
					}
				}
			}
		}
	}

	return result;
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
