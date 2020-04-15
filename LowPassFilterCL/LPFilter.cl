__kernel void LPFilter(
	__global uint* input, 
	__global uint* output, 
	__local uint* row, 
	__local uint* outRow,
	uint width
) {
	size_t i = get_global_id(0);

	if (i >= get_global_size(0)) {
		return;
	}

	// The current working row number
	size_t currentRow = i / width;
	// The item index in the current row
	size_t localId = i - (currentRow * width);
	
	// Copy row item from global memory to local memory
	if (localId == 0 && i > 0) {
		row[localId] = (input[i] + input[i - 1]) / 2;
	} else {
		row[localId] = input[i];
	}

	// Wait for all work items in the group to finish copying before executing further
	barrier(CLK_LOCAL_MEM_FENCE);
	
	// Perform Low Pass Filter
	if (localId > 0) {
		outRow[localId] = (row[localId] + row[localId - 1]) / 2;
	} else {
		outRow[localId] = row[localId];
	}

	// Copy row item back to global memory
	output[i] = outRow[localId];
}