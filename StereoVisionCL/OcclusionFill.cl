__kernel void OcclusionFill(
	__global uint* map,
	__global uint* result,
	uint width,
	uint height,
	int occlusionNeighbours
) {
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	uint currentIndex = i * width + j;
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