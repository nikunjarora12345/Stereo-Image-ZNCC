__kernel void AddMatrices(__global int* a, __global int* b, __global int* c) {
	size_t x = get_global_id(0);
	
	c[x] = a[x] + b[x];
}