__kernel void CrossCheck(
	__global uint* leftDisp, 
	__global uint* rightDisp, 
	__global uint* result, 
	int crossCheckingThreshold
) {
	size_t i = get_global_id(0);

	int diff = leftDisp[i] - rightDisp[i];
	if (diff >= 0) { // leftDisp is winner
		if (diff <= crossCheckingThreshold) {
			result[i] = leftDisp[i];
		}
		else {
			result[i] = 0;
		}
	} else { //  rightDisp is winner
		if (-diff <= crossCheckingThreshold) {
			result[i] = rightDisp[i];
		}
		else {
			result[i] = 0;
		}

	}
}