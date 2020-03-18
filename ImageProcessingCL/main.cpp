#include <CL/cl.hpp>
#include <iostream>
#include <fstream>
#include <chrono>

#include "lodepng.h"

int main() {
	// Time the execution of entire program
	std::chrono::high_resolution_clock::time_point start, end;
	unsigned long duration;

	start = std::chrono::high_resolution_clock::now();

	// The array containing pixel values in RGBARGBA... format
	std::vector<unsigned char> pixels;
	
	unsigned width, height;
	unsigned windowSize = 5; // moving window size (windowSize * windowSize)

	unsigned error = lodepng::decode(pixels, width, height, "input.png");
	if (error) {
		std::cout << "Error reading the image: " << error << std::endl;
		return -1;
	}

	// Output array
	std::vector<unsigned char> output(pixels.size());
	// Grayscale array
	std::vector<unsigned char> grayscale(width * height);

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
	std::ifstream fileStream("ProcessImg.cl");
	std::string src(std::istreambuf_iterator<char>(fileStream), (std::istreambuf_iterator<char>()));

	// Load the kernel code
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	cl::Context context(device);
	cl::Program program(context, sources);

	cl_int err = program.build("-cl-std=CL1.2");

	// Split the input image into RGB array to make parallel computing possible
	std::vector<unsigned char> r(width * height);
	std::vector<unsigned char> g(width * height);
	std::vector<unsigned char> b(width * height);

	int index = 0;
	for (int i = 0; i < pixels.size(); i += 4) {
		r[index] = pixels[i];
		g[index] = pixels[i + 1];
		b[index] = pixels[i + 2];

		index++;
	}

	cl::Buffer inRBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(r[0]) * r.size(), r.data(), &err);
	cl::Buffer inGBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(g[0]) * g.size(), g.data(), &err);
	cl::Buffer inBBuff(context, CL_MEM_READ_ONLY | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
		sizeof(b[0]) * b.size(), b.data(), &err);
	cl::Buffer grayBuff(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(grayscale[0]) * grayscale.size(), grayscale.data(), &err);

	// Run the Grayscale function. Wait for it to finish, then run the Average function
	cl::Kernel kernelGray(program, "Rgb2Gray");
	err = kernelGray.setArg(0, inRBuff);
	err = kernelGray.setArg(1, inGBuff);
	err = kernelGray.setArg(2, inBBuff);
	err = kernelGray.setArg(3, grayBuff);

	// Event for profiling the execution time of kernel
	cl::Event grayEvent;
	cl::Event avgEvent;

	cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
	err = queue.enqueueNDRangeKernel(kernelGray, cl::NullRange, cl::NDRange(width * height),
		cl::NullRange, NULL, &grayEvent);
	err = queue.enqueueReadBuffer(grayBuff, CL_TRUE, 0, sizeof(grayscale[0]) * grayscale.size(), grayscale.data());

	grayEvent.wait();
	unsigned elapsed = grayEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>(&err) -
		grayEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>(&err);

	std::cout << "The Grayscale function took " << elapsed / 1000 << " microseconds." << std::endl;

	// Now we run the Average function
	cl::Kernel kernelAvg(program, "AverageFilter");
	err = kernelAvg.setArg(0, grayBuff);
	err = kernelAvg.setArg(1, width);
	err = kernelAvg.setArg(2, windowSize);

	err = queue.enqueueNDRangeKernel(kernelAvg, cl::NullRange, cl::NDRange(width * height), 
		cl::NullRange, NULL, &avgEvent);
	err = queue.enqueueReadBuffer(grayBuff, CL_TRUE, 0, sizeof(grayscale[0]) * grayscale.size(), grayscale.data());

	avgEvent.wait();
	elapsed = avgEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>(&err) -
		avgEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>(&err);

	std::cout << "The Average function took " << elapsed / 1000 << " microseconds." << std::endl;

	// Fill output array with average values
	index = 0;
	for (int i = 0; i < output.size(); i+=4) {
		output[i] = output[i + 1] = output[i + 2] = grayscale[index++];
		output[i + 3] = pixels[i + 3];
	}

	if (err) {
		std::cout << err << std::endl;
		std::cin.get();
		return -1;
	}

	error = lodepng::encode("output.png", output, width, height);

	if (error) {
		std::cout << "Error: " << error << ": " << lodepng_error_text(error) << std::endl;
	}

	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
	std::cout << "The entire program took " << duration << " microseconds." << std::endl;

	std::cin.get();

	return 0;
}