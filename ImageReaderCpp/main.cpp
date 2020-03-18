#include "lodepng.h"

#include <iostream>

int main() {
	// The pixels array containing image in the form RGBARGBA...
	std::vector<unsigned char> pixels;
	unsigned width, height;

	// Decode from filename
	unsigned error = lodepng::decode(pixels, width, height, "input.png");

	if (error) {
		std::cout << "Error: " << error << ": " << lodepng_error_text(error) << std::endl;
	}

	/*
	Convert RGB image to grayscale using the formula:
	Grayscale = (0.3 * R) + (0.59 * G) + (0.11 * B)
	Add the alpha value as it is
	*/
	std::vector<unsigned char> grayscalePixels;
	unsigned char grayscaleValue = 0;
	for (int i = 0; i < pixels.size(); i++) {
		if (i % 4 == 0) {
			// Red value
			grayscaleValue += 0.3 * pixels[i];
		} else if (i % 4 == 1) {
			// Green value
			grayscaleValue += 0.59 * pixels[i];
		} else if (i % 4 == 2) {
			// Blue value
			grayscaleValue += 0.11 * pixels[i];
		} else if (i % 4 == 3) {
			// Alpha value

			// Set pixel value to be 0 if below 128
			if (grayscaleValue < 128) {
				grayscaleValue = 0;
			}

			// Add the pixel values to the array in the form of RGBARGBA...
			grayscalePixels.push_back(grayscaleValue); // R
			grayscalePixels.push_back(grayscaleValue); // G
			grayscalePixels.push_back(grayscaleValue); // B
			grayscalePixels.push_back(pixels[i] < 128 ? 0 : pixels[i]); // A

			grayscaleValue = 0;
		}
	}

	error = lodepng::encode("output.png", grayscalePixels, width, height);

	if (error) {
		std::cout << "Error: " << error << ": " << lodepng_error_text(error) << std::endl;
	}
}