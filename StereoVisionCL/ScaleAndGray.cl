__kernel void ScaleAndGray(__global uchar* orig, __global uint* gray, uint width, uint height, int scaleFactor) {
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	int newWidth = width / scaleFactor;
	
	int x = (scaleFactor * i - 1 * (i > 0));
	int y = (scaleFactor * j - 1 * (j > 0));

	gray[i * newWidth + j] = 
		0.3 * orig[x * (4 * width) + 4 * y] + 
		0.59 * orig[x * (4 * width) + 4 * y + 1] + 
		0.11 * orig[x * (4 * width) + 4 * y + 2];
}