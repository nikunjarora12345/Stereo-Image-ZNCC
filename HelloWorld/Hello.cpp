#include <CL/cl.hpp>
#include <iostream>
#include <fstream>

int main() {
	// Get the list of platforms available
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	// Get the list of devices from the first platform
	cl::Platform platform = platforms.front();
	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cl::Device device = devices.front();
	
	// Read the kernel code
	std::ifstream HelloWorldFile("HelloWorld.cl");
	std::string src(std::istreambuf_iterator<char>(HelloWorldFile), (std::istreambuf_iterator<char>()));

	// Load the kernel code
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.length() + 1));

	cl::Context context(device);
	cl::Program program(context, sources);

	cl_int err = program.build("-cl-std=CL1.2");

	// Create the char array and call the kernel function
	char buff[16];
	cl::Buffer memBuff(context, CL_MEM_WRITE_ONLY | CL_MEM_HOST_READ_ONLY, sizeof(buff));
	cl::Kernel kernel(program, "HelloWorld", &err);
	kernel.setArg(0, memBuff);

	cl::CommandQueue queue(context, device);
	queue.enqueueTask(kernel);
	queue.enqueueReadBuffer(memBuff, CL_TRUE, 0, sizeof(buff), buff);

	std::cout << buff << std::endl;

	std::cin.get();
}