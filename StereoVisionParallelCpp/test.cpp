#include <iostream>
#include <chrono>
#include <omp.h>

class Timer {
private:
	std::chrono::time_point<std::chrono::steady_clock> start, end;
	std::chrono::duration<float> duration;

public:
	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;

		std::cout << "Done (" << duration.count() << " s)" << std::endl;
	}

	float getElapsedTime() {
		end = std::chrono::high_resolution_clock::now();
		duration = end - start;

		return duration.count();
	}
};

static long numSteps = 100000;
double step;

double calculatePiSeries() {
	Timer timer;

	double sum = 0.0;

	step = 1.0 / (double)numSteps;

	for (int i = 0; i < numSteps; i++) {
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}
	return step * sum;
}

double calculatePiParallel() {
	Timer timer;

	omp_set_num_threads(4);
	
	double sum = 0.0;

	step = 1.0 / (double)numSteps;

	#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < numSteps; i++) {
		double x = (i + 0.5) * step;
		sum += 4.0 / (1.0 + x * x);
	}
	return sum * step;
}

int _main() {
	std::cout << calculatePiSeries() << std::endl;
	std::cout << calculatePiParallel() << std::endl;

	std::cin.get();
	return 0;
}