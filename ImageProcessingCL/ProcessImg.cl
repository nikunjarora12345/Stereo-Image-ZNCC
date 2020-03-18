__kernel void Rgb2Gray(__global uchar* r, __global uchar* g, __global uchar* b, __global uchar* gray) {
	size_t gid = get_global_id(0);

	// Convert to grayscale
	gray[gid] = (0.3 * r[gid]) + (0.59 * g[gid]) + (0.11 * b[gid]);
}

__kernel void AverageFilter(__global uchar* gray, uint width, uint windowSize) {
	size_t gid = get_global_id(0);

	uint sum = 0;
	// -1 because it is 0 based indexing
	uint windowMedian = ((windowSize + 1) / 2) - 1;

	// Controls the column of moving window
	for(uint i = gid; i < gid + windowSize; i++) {
		uint currentIndex = i;
			
		// Controls the row of moving window
		for(uint k = 0; k < windowSize; k++) {
			currentIndex += width * k;
			sum += gray[currentIndex];
		}
	}

	// 1D-2D mapping (i = x + width*y)
	gray[(gid + windowMedian) + width * windowMedian] = sum / (windowSize * windowSize);
}