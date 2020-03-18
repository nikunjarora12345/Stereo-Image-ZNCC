#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>

// Function Prototypes
int** AddCpp(int**, int**);
int** AddCl(int**, int**);

const int NUM_ROWS = 100;
const int NUM_COLS = 100;

int main() {
	// Seed the random value generator
	std::srand(1);

	int** a = new int* [NUM_ROWS];
	int** b = new int* [NUM_ROWS];

	for (int i = 0; i < NUM_ROWS; i++) {
		a[i] = new int[NUM_COLS];
		b[i] = new int[NUM_COLS];

		for (int j = 0; j < NUM_COLS; j++) {
			// Assign random value between 0 and 100
			a[i][j] = std::rand() % 100;
			b[i][j] = std::rand() % 100;
		}
	}

	// For measuring time
	std::chrono::high_resolution_clock::time_point start, end;
	long duration = 0;

	int** cl = AddCl(a, b);

	start = std::chrono::high_resolution_clock::now();
	int** cpp = AddCpp(a, b);
	end = std::chrono::high_resolution_clock::now();

	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "C++ Function took " << duration << " microseconds." << std::endl;

	// Write the two outputs to csv files
	std::ofstream cppFile("cpp.csv", std::ios::out | std::ios::trunc);
	std::ofstream clFile("cl.csv", std::ios::out | std::ios::trunc);

	for (int i = 0; i < NUM_ROWS; i++) {
		for (int j = 0; j < NUM_COLS; j++) {
			cppFile << cpp[i][j] << ",";
			clFile << cl[i][j] << ",";
		}

		cppFile << "\n";
		clFile << "\n";
	}

	std::cin.get();

	// Free the memories
	cppFile.close();
	clFile.close();

	for (int i = 0; i < NUM_ROWS; i++) {
		delete[] a[i];
		delete[] b[i];
		delete[] cpp[i];
		delete[] cl[i];
	}

	delete[] a;
	delete[] b;
	delete[] cpp;
	delete[] cl;
}

// Add two matrices using plain c++
int** AddCpp(int** a, int** b) {
	int** c = new int* [NUM_ROWS];

	for (int i = 0; i < NUM_ROWS; i++) {
		c[i] = new int[NUM_COLS];

		for (int j = 0; j < NUM_COLS; j++) {
			c[i][j] = a[i][j] + b[i][j];
		}
	}

	return c;
}

// Add two matrices using opencl
int** AddCl(int** a, int** b) {
	// Get the list of platforms available
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	// Get the list of devices from the first platform
	cl::Platform platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cl::Device device = devices.front();

	std::cout << "Device: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
	std::cout << "OpenCL Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;

	// Read the kernel code
	std::ifstream fileStream("AddMatrices.cl");
	std::string src(std::istreambuf_iterator<char>(fileStream), (std::istreambuf_iterator<char>()));

	// Load the kernel code
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	cl::Context context(device);
	cl::Program program(context, sources);

	cl_int err = program.build("-cl-std=CL1.2");

	// Flatten out the matrices
	int* aFlat = new int[NUM_ROWS * NUM_COLS];
	int* bFlat = new int[NUM_ROWS * NUM_COLS];
	int** c = new int* [NUM_ROWS]; // Output Matrix
	int* cFlat = new int[NUM_ROWS * NUM_COLS];

	int k = 0;
	for (int i = 0; i < NUM_ROWS; i++) {
		c[i] = new int[NUM_COLS];

		for (int j = 0; j < NUM_COLS; j++) {
			aFlat[k] = a[i][j];
			bFlat[k] = b[i][j];

			k++;
		}
	}

	// For measuring time
	std::chrono::high_resolution_clock::time_point start, end;
	long duration = 0;

	start = std::chrono::high_resolution_clock::now();

	cl::Buffer aBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		NUM_COLS * NUM_ROWS * sizeof(int), aFlat);
	cl::Buffer bBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		NUM_COLS * NUM_ROWS * sizeof(int), bFlat);
	cl::Buffer cBuff(context, CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		NUM_COLS * NUM_ROWS * sizeof(int), cFlat);

	cl::Kernel kernel(program, "AddMatrices");
	kernel.setArg(0, aBuff);
	kernel.setArg(1, bBuff);
	kernel.setArg(2, cBuff);

	cl::CommandQueue queue(context, device);
	queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(NUM_ROWS * NUM_COLS * 2));
	queue.enqueueReadBuffer(cBuff, CL_TRUE, 0, NUM_COLS * NUM_ROWS * sizeof(int), cFlat);

	end = std::chrono::high_resolution_clock::now();

	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "OpenCL Kernel took " << duration << " microseconds." << std::endl;

	// Fill Matrix A from the flattened result
	k = 0;
	for (int i = 0; i < NUM_ROWS; i++) {
		for (int j = 0; j < NUM_COLS; j++) {
			c[i][j] = cFlat[k++];
		}
	}

	delete[] aFlat;
	delete[] bFlat;
	delete[] cFlat;

	return c;
}