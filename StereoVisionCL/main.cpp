#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cassert>

#include "lodepng.h"

cl_int err;

// Efficient error handling
#define CLCall(x) \
	err = x; \
	if (err) \
		std::cout << "Error [" << err << "]: " << __FILE__ << ":" << __LINE__ << std::endl

/*
Class to handle multiple opencl kernel files
*/
class CLProgram {
private:
	cl::Program m_Program;
	
public:
	CLProgram(cl::Context context, cl::Device device, const std::string& fileName) {
		// Read the kernel codes
		std::ifstream fileStream(fileName);
		std::string src(std::istreambuf_iterator<char>(fileStream), (std::istreambuf_iterator<char>()));

		// Load the kernel code
		cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

		m_Program = cl::Program(context, sources);

		CLCall(m_Program.build("-cl-std=CL1.2"));

		cl_build_status status = m_Program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
		if (status == CL_BUILD_ERROR) {
			std::string log = m_Program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			std::cerr << log << std::endl;
		}
	}

	inline cl::Program GetProgram() { return m_Program; }
};

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

constexpr int maxDisparity = 64;

constexpr int windowWidth = 15;
constexpr int windowHeight = 15;

constexpr int crossCheckingThreshold = 2;

constexpr int occlusionNeighbours = 256;

constexpr int scaleFactor = 4;

std::vector<unsigned char> loadImage(const char*, unsigned&, unsigned&);
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

	std::vector<unsigned char> leftPixels, rightPixels;
	unsigned width, height, rightWidth, rightHeight;

	std::cout << "Reading Left Image...";
	leftPixels = loadImage("imageL.png", width, height);

	std::cout << "Reading Right Image...";
	rightPixels = loadImage("imageR.png", rightWidth, rightHeight);

	// left and right images are assumed to be of same dimensions
	assert(width == rightWidth && height == rightHeight);

	width /= scaleFactor;
	height /= scaleFactor;
	unsigned imgSize = width * height;
	
	// Create Programs
	CLProgram scaleProg(context, device, "ScaleAndGray.cl");
	CLProgram znccProg(context, device, "Zncc.cl");
	CLProgram crossCheckProg(context, device, "CrossCheck.cl");
	CLProgram ocFillProg(context, device, "OcclusionFill.cl");
	
	// Array to copy output back into
	std::vector<unsigned> output(imgSize);

	// Create buffers
	cl::Buffer lBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR, 
		sizeof(leftPixels[0]) * leftPixels.size(), leftPixels.data());
	cl::Buffer rBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(rightPixels[0]) * rightPixels.size(), rightPixels.data());
	cl::Buffer grayLBuff(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(output[0]) * imgSize, output.data());
	cl::Buffer grayRBuff(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(output[0]) * imgSize, output.data());
	cl::Buffer dispLRBuff(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(output[0]) * imgSize, output.data());
	cl::Buffer dispRLBuff(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(output[0]) * imgSize, output.data());
	cl::Buffer dispCCBuff(context, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(output[0]) * imgSize, output.data());
	cl::Buffer outputBuff(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(output[0]) * imgSize, output.data());

	// Create Kernels
	cl::Kernel scaleKernel(scaleProg.GetProgram(), "ScaleAndGray");
	CLCall(scaleKernel.setArg(0, lBuff));
	CLCall(scaleKernel.setArg(1, rBuff));
	CLCall(scaleKernel.setArg(2, grayLBuff));
	CLCall(scaleKernel.setArg(3, grayRBuff));
	CLCall(scaleKernel.setArg(4, width * scaleFactor));
	CLCall(scaleKernel.setArg(5, height * scaleFactor));
	CLCall(scaleKernel.setArg(6, scaleFactor));

	cl::Kernel dispKernel(znccProg.GetProgram(), "Zncc");
	CLCall(dispKernel.setArg(0, grayLBuff));
	CLCall(dispKernel.setArg(1, grayRBuff));
	CLCall(dispKernel.setArg(2, dispLRBuff));
	CLCall(dispKernel.setArg(3, dispRLBuff));
	CLCall(dispKernel.setArg(4, width));
	CLCall(dispKernel.setArg(5, height));
	CLCall(dispKernel.setArg(6, 0));
	CLCall(dispKernel.setArg(7, maxDisparity));
	CLCall(dispKernel.setArg(8, windowWidth));
	CLCall(dispKernel.setArg(9, windowHeight));

	cl::Kernel dispCCKernel(crossCheckProg.GetProgram(), "CrossCheck");
	CLCall(dispCCKernel.setArg(0, dispLRBuff));
	CLCall(dispCCKernel.setArg(1, dispRLBuff));
	CLCall(dispCCKernel.setArg(2, dispCCBuff));
	CLCall(dispCCKernel.setArg(3, crossCheckingThreshold));

	cl::Kernel ocFillKernel(ocFillProg.GetProgram(), "OcclusionFill");
	CLCall(ocFillKernel.setArg(0, dispCCBuff));
	CLCall(ocFillKernel.setArg(1, outputBuff));
	CLCall(ocFillKernel.setArg(2, width));
	CLCall(ocFillKernel.setArg(3, height));
	CLCall(ocFillKernel.setArg(4, occlusionNeighbours));

	// Events
	double elapsed = 0;

	cl::Event scaleLEvent;
	cl::Event scaleREvent;
	cl::Event dispLREvent;
	cl::Event dispRLEvent;
	cl::Event dispCCEvent;
	cl::Event ocFillEvent;

	// Enqueue Tasks
	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Scale and gray left
	std::cout << "Converting Images to grayscale...";
	CLCall(queue.enqueueNDRangeKernel(scaleKernel, cl::NullRange, 
		cl::NDRange(height, width), cl::NullRange, nullptr, &scaleLEvent));
	scaleLEvent.wait();
	elapsed = scaleLEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		scaleLEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	std::cout << "Done (" << elapsed * 1e-9 << " s)" << std::endl;

	// Disparity Maps
	std::cout << "Calculating Disparity Maps...";
	CLCall(queue.enqueueNDRangeKernel(dispKernel, cl::NullRange,
		cl::NDRange(height, width), cl::NDRange(2, 15), nullptr, &dispLREvent));
	dispLREvent.wait();
	elapsed = dispLREvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		dispLREvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	std::cout << "Done (" << elapsed * 1e-9 << " s)" << std::endl;

	// Cross Checking
	std::cout << "Performing Cross Checking...";
	CLCall(queue.enqueueNDRangeKernel(dispCCKernel, cl::NullRange, 
		cl::NDRange(height * width), cl::NullRange, nullptr, &dispCCEvent));
	dispCCEvent.wait();
	elapsed = dispCCEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		dispCCEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	std::cout << "Done (" << elapsed * 1e-9 << " s)" << std::endl;

	// Occlusion Filling
	std::cout << "Performing Occlusion Filling...";
	CLCall(queue.enqueueNDRangeKernel(ocFillKernel, cl::NullRange,
		cl::NDRange(height, width), cl::NullRange, nullptr, &ocFillEvent));
	ocFillEvent.wait();
	elapsed = ocFillEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		ocFillEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	std::cout << "Done (" << elapsed * 1e-9 << " s)" << std::endl;

	// Read output
	CLCall(cl::enqueueReadBuffer(outputBuff, CL_TRUE, 0, sizeof(output[0])* output.size(), output.data()));
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

	// Normalize values to be between 0 and 255
	int mapIndex = 0;
	for (int i = 0; i < width * height * 4; i += 4, mapIndex++) {
		result[i] = result[i + 1] = result[i + 2] = (unsigned char)(255 * (in[mapIndex] - min) / (max - min));
		result[i + 3] = 255;
	}

	return result;
}