#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

#include "lodepng.h"

cl_int err;

// Efficient error handling
#define CLCall(x) \
	err = x; \
	if (err) \
		std::cout << "Error [" << err << "]: " << __FILE__ << ":" << __LINE__ << std::endl

/*
Class to calculate time taken by functions in seconds.
* Creating an object of the class in a function, calls the constructor which starts the timer.
* At the end of the function, the destructor is called which stops the timer and calculates the duration.
* We can get the duration manually using the getElapsedTime method.
*/
class Timer {
private:
	std::chrono::time_point<std::chrono::steady_clock> m_Start, m_End;
	std::chrono::duration<float> m_Duration;

public:
	Timer() {
		m_Start = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		m_End = std::chrono::high_resolution_clock::now();
		m_Duration = m_End - m_Start;

		std::cout << "Done (" << m_Duration.count() << " s)" << std::endl;
	}

	float getElapsedTime() {
		m_End = std::chrono::high_resolution_clock::now();
		m_Duration = m_End - m_Start;

		return m_Duration.count();
	}
};

std::vector<unsigned char> loadImage(const char*, unsigned&, unsigned&);
std::vector<unsigned> denormalize(std::vector<unsigned char>, const unsigned, const unsigned);
std::vector<unsigned char> normalize(std::vector<unsigned>, const unsigned, const unsigned);

int main() {
	Timer timer;

	// Get the list of platforms available
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	// Get the list of devices from the first platform
	cl::Platform platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cl::Device device = devices.front();
	cl::Context context(device);

	// Get the maximum number of work items per work group supported by the GPU
	unsigned maxWorkGroupSize = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

	std::cout << "Device: " << device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
	std::cout << "OpenCL Version: " << device.getInfo<CL_DEVICE_VERSION>() << std::endl;
	std::cout << "Max Workgroup Size: " << maxWorkGroupSize << std::endl;
	std::cout << "Max Local Memory Size: " << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
	std::cout << std::endl;

	// Read the kernel codes
	std::ifstream fileStream("LPFilter.cl");
	std::string src(std::istreambuf_iterator<char>(fileStream), (std::istreambuf_iterator<char>()));

	// Load the kernel code
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	cl::Program program(context, sources);

	CLCall(program.build("-cl-std=CL1.2"));


	std::vector<unsigned char> originalPixels;
	std::vector<unsigned> input;
	unsigned width, height;

	std::cout << "Reading Input Image...";
	originalPixels = loadImage("input.png", width, height);
	input = denormalize(originalPixels, width, height);

	unsigned imSize = width * height;
	unsigned factor = 1;

	// Find the highest number less than maximum work group size which can divide the image
	for (int i = maxWorkGroupSize; i >= 1; i--) {
		if (imSize % i == 0) {
			factor = i;
			break;
		}
	}

	std::vector<unsigned> output(imSize);

	cl::Buffer inBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(input[0]) * input.size(), input.data());
	cl::Buffer outBuff(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(output[0]) * output.size(), output.data());

	cl::Kernel kernel(program, "LPFilter");
	kernel.setArg(0, inBuff);
	kernel.setArg(1, outBuff);
	kernel.setArg(2, sizeof(unsigned) * factor);
	kernel.setArg(3, sizeof(unsigned) * factor);
	kernel.setArg(4, factor);

	cl::Event ev;
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	std::cout << "Running OpenCL Kernel...";

	// The image is split in arbitrary widths of size "factor"
	CLCall(queue.enqueueNDRangeKernel(kernel, cl::NullRange,
		cl::NDRange(height * width), cl::NDRange(factor), nullptr, &ev));
	ev.wait();
	double elapsed = ev.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		ev.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	std::cout << "Done (" << elapsed * 1e-9 << " s)" << std::endl;

	CLCall(cl::enqueueReadBuffer(outBuff, CL_TRUE, 0, sizeof(output[0]) * output.size(), output.data()));
	lodepng::encode("output.png", normalize(output, width, height), width, height);

	std::cout << "The program took " << timer.getElapsedTime() << " s" << std::endl;

	std::cin.get();
	return 0;
}

std::vector<unsigned char> loadImage(const char* filename, unsigned& width, unsigned& height) {
	Timer timer;

	std::vector<unsigned char> pixels;

	unsigned error = lodepng::decode(pixels, width, height, filename);
	if (error) {
		std::cout << "Failed to load image: " << lodepng_error_text(error) << std::endl;
		std::cin.get();
		exit(-1);
	}

	return pixels;
}

// Convert RGBA array to grayscale pixels array
std::vector<unsigned> denormalize(
	std::vector<unsigned char> in,
	const unsigned width,
	const unsigned height
) {
	std::vector<unsigned> result(width * height);

	for (int i = 0, index = 0; i < width * height * 4; i += 4, index++) {
		result[index] = in[i];
	}

	return result;
}

// Convert grayscale pixels array to RGBA array
std::vector<unsigned char> normalize(
	std::vector<unsigned> in,
	const unsigned width,
	const unsigned height
) {
	std::vector<unsigned char> result(width * height * 4);

	unsigned char max = 0;
	unsigned char min = UCHAR_MAX;

	for (int i = 0; i < width * height; i++) {
		if (in[i] > max) {
			max = in[i];
		}

		if (in[i] < min) {
			min = in[i];
		}
	}

	int mapIndex = 0;
	for (int i = 0; i < width * height * 4; i += 4, mapIndex++) {
		result[i] = result[i + 1] = result[i + 2] = (unsigned char)(255 * (in[mapIndex] - min) / (max - min));
		result[i + 3] = 255;
	}

	return result;
}
