__kernel void Zncc(
	__global uint* leftPixels, 
	__global uint* rightPixels, 
	__global uint* dispLRMap, 
	__global uint* dispRLMap, 
	uint width, 
	uint height, 
	int minDisp, 
	int maxDisp,
	int windowWidth,
	int windowHeight
) {
	uint windowSize = windowWidth * windowHeight;

	size_t gi = get_global_id(0);
	size_t gj = get_global_id(1);

	size_t h = get_global_size(0);
	size_t w = get_global_size(1);

	float l_bestDisparity = maxDisp;
	float l_bestZncc = -1;
	float r_bestDisparity = minDisp;
	float r_bestZncc = -1;

	for (int i = gi; i < height; i += h) {
		for (int j = gj; j < width; j += w) {
			// Select the best disparity value for the current pixel
			for (int d = minDisp; d <= maxDisp; d++) {
				int l_d = d;
				int r_d = -d;

				// Calculating mean of blocks using the sliding window method
				float l_meanLBlock = 0, l_meanRBlock = 0, r_meanLBlock = 0, r_meanRBlock = 0;

				for (int x = -windowHeight / 2; x < windowHeight / 2; x++) {
					for (int y = -windowWidth / 2; y < windowWidth / 2; y++) {
						// Check for image borders
						if (
							!(i + x >= 0) ||
							!(i + x < height) ||
							!(j + y >= 0) ||
							!(j + y < width) ||
							!(j + y - l_d >= 0) ||
							!(j + y - r_d >= 0) ||
							!(j + y - l_d < width) ||
							!(j + y - r_d < width)
							) {
							continue;
						}

						l_meanLBlock += leftPixels[(i + x) * width + (j + y)];
						l_meanRBlock += rightPixels[(i + x) * width + (j + y - l_d)];
						r_meanLBlock += rightPixels[(i + x) * width + (j + y)];
						r_meanRBlock += leftPixels[(i + x) * width + (j + y - r_d)];
					}
				}

				l_meanLBlock /= windowSize;
				l_meanRBlock /= windowSize;
				r_meanLBlock /= windowSize;
				r_meanRBlock /= windowSize;

				// Calculate ZNCC for current disparity value
				float l_stdLBlock = 0, l_stdRBlock = 0, r_stdLBlock = 0, r_stdRBlock = 0;
				float l_currentZncc = 0, r_currentZncc = 0;

				for (int x = -windowHeight / 2; x < windowHeight / 2; x++) {
					for (int y = -windowWidth / 2; y < windowWidth / 2; y++) {
						// Check for image borders
						if (
							!(i + x >= 0) ||
							!(i + x < height) ||
							!(j + y >= 0) ||
							!(j + y < width) ||
							!(j + y - l_d >= 0) ||
							!(j + y - r_d >= 0) ||
							!(j + y - l_d < width) ||
							!(j + y - r_d < width)
							) {
							continue;
						}

						int l_centerL = leftPixels[(i + x) * width + (j + y)] - l_meanLBlock;
						int l_centerR = rightPixels[(i + x) * width + (j + y - l_d)] - l_meanRBlock;
						int r_centerL = rightPixels[(i + x) * width + (j + y)] - r_meanLBlock;
						int r_centerR = leftPixels[(i + x) * width + (j + y - r_d)] - r_meanRBlock;

						// standard deviation
						l_stdLBlock += l_centerL * l_centerL;
						l_stdRBlock += l_centerR * l_centerR;

						l_currentZncc += l_centerL * l_centerR;

						r_stdLBlock += r_centerL * r_centerL;
						r_stdRBlock += r_centerR * r_centerR;

						r_currentZncc += r_centerL * r_centerR;
					}
				}

				l_currentZncc /= native_sqrt(l_stdLBlock) * native_sqrt(l_stdRBlock);
				r_currentZncc /= native_sqrt(r_stdLBlock) * native_sqrt(r_stdRBlock);

				// Selecting best disparity
				if (l_currentZncc > l_bestZncc) {
					l_bestZncc = l_currentZncc;
					l_bestDisparity = l_d;
				}
				if (r_currentZncc > r_bestZncc) {
					r_bestZncc = r_currentZncc;
					r_bestDisparity = r_d;
				}
			}
		}
	}

	dispLRMap[gi * width + gj] = (uint) fabs(l_bestDisparity);
	dispRLMap[gi * width + gj] = (uint) fabs(r_bestDisparity);
}