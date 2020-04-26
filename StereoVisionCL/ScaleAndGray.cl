__kernel void ScaleAndGray(
	__global uchar* lOrig,
	__global uchar* rOrig,
	__global uint* lGray, 
	__global uint* rGray, 
	uint width, 
	uint height, 
	int scaleFactor
) {
	size_t i = get_global_id(0);
	size_t j = get_global_id(1);

	int newWidth = width / scaleFactor;
	
	int x = (scaleFactor * i - 1 * (i > 0));
	int y = (scaleFactor * j - 1 * (j > 0));

	lGray[i * newWidth + j] = 
		0.3 * lOrig[x * (4 * width) + 4 * y] + 
		0.59 * lOrig[x * (4 * width) + 4 * y + 1] + 
		0.11 * lOrig[x * (4 * width) + 4 * y + 2];

	rGray[i * newWidth + j] = 
		0.3 * rOrig[x * (4 * width) + 4 * y] + 
		0.59 * rOrig[x * (4 * width) + 4 * y + 1] + 
		0.11 * rOrig[x * (4 * width) + 4 * y + 2];
}