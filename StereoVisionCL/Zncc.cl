__kernel void Zncc(
	__global uint* leftPixels, 
	__global uint* rightPixels, 
	__global uint* disparityMap,
	uint width, 
	uint height, 
	int minDisp, 
	int maxDisp,
	int windowWidth,
	int windowHeight
) {
	uint windowSize = windowWidth * windowHeight;

	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

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

		currentZncc /= native_sqrt(stdLBlock) * native_sqrt(stdRBlock);

		// Selecting best disparity
		if (currentZncc > bestZncc) {
			bestZncc = currentZncc;
			bestDisparity = d;
		}
	}

	disparityMap[i * width + j] = (uint) fabs(bestDisparity);
}