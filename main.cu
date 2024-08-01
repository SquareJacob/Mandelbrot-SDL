#include <SDL.h>
#include <SDL_image.h>
#include <SDL_ttf.h>
#include <SDL_mixer.h>
#include <iostream>
#include <stdlib.h>  
#include <crtdbg.h>   //for malloc and free
#include <set>
#include <math.h>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#define _CRTDBG_MAP_ALLOC
#ifdef _DEBUG
#define new new( _NORMAL_BLOCK, __FILE__, __LINE__)
#endif

typedef std::chrono::steady_clock::time_point tp;
typedef std::chrono::microseconds ms;
tp now() {
	return std::chrono::high_resolution_clock::now();
}
struct cell {
	double frac;
	int iter;
};

SDL_Window* window;
SDL_Renderer* renderer;
bool running;
SDL_Event event;
std::set<std::string> keys;
std::set<std::string> currentKeys;
int mouseX = 0;
int mouseY = 0;
int mouseDeltaX, mouseDeltaY;
int mouseScroll = 0;
std::set<int> buttons;
std::set<int> currentButtons;
tp start, trueStart;
int calcTime, drawTime;

double topX = -2.0, topY = -1.5, zoom = 200.0, zoomFactor, localMouseX, localMouseY, color, sReal = 0.0, sImg = 0.0;
cell pixels[600 * 800];
cell* d_pixels;
int maxIter = 1000;
cell pixel;
bool timing = true;
int totalTime = 60;

void debug(int line, std::string file) {
	std::cout << "Line " << line << " in file " << file << ": " << SDL_GetError() << std::endl;
}

__global__ void mandelbrot(double topX, double topY, double delta, int max, cell pixels[600 * 800], double sReal, double sImg) {
	double localX = topX + static_cast<float>(blockIdx.x) * delta;
	double localY = topY + static_cast<float>(threadIdx.x) * delta;
	/**double squared = localX * localX + localY * localY;
	if (64.0 * squared * squared - 24.0 * squared < 0.75 - 8 * localX || squared < -0.9375 - 2 * localX) {
		pixels[threadIdx.x * 800 + blockIdx.x] = { 0.0, max };
		return;
	}
	**/
	int i;
	double real = sReal, tmpReal, img = sImg;
	for (i = 0; i < max; i++) {
		tmpReal = real;
		real = real * real - img * img + localX;
		img = 2.0 * tmpReal * img + localY;
		if (real * real + img * img > 400.0) {
			break;
		}
	}
	pixels[threadIdx.x * 800 + blockIdx.x] = { log2(0.5 * log(img * img + real * real) / log(20.0)), i};
}

int main(int argc, char* argv[]) {
	if (SDL_Init(SDL_INIT_EVERYTHING) == 0 && TTF_Init() == 0 && Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) == 0) {
		//Setup
		window = SDL_CreateWindow("Window", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 800, 600, 0);
		if (window == NULL) {
			debug(__LINE__, __FILE__);
			return 0;
		}

		renderer = SDL_CreateRenderer(window, -1, 0);
		if (renderer == NULL) {
			debug(__LINE__, __FILE__);
			return 0;
		}
		SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_NONE);

		SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA8888,
			SDL_TEXTUREACCESS_STREAMING, 800, 600);
		void* txtPixels;
		int pitch;
		SDL_PixelFormat* format = SDL_AllocFormat(SDL_PIXELFORMAT_RGBA8888);

		//Main loop
		running = true;
		while (running) {
			//handle events
			for (std::string i : keys) {
				currentKeys.erase(i); //make sure only newly pressed keys are in currentKeys
			}
			for (int i : buttons) {
				currentButtons.erase(i); //make sure only newly pressed buttons are in currentButtons
			}
			mouseScroll = 0;
			mouseDeltaX = 0;
			mouseDeltaY = 0;
			while (SDL_PollEvent(&event)) {
				switch (event.type) {
				case SDL_QUIT:
					running = false;
					break;
				case SDL_KEYDOWN:
					if (!keys.contains(std::string(SDL_GetKeyName(event.key.keysym.sym)))) {
						currentKeys.insert(std::string(SDL_GetKeyName(event.key.keysym.sym)));
					}
					keys.insert(std::string(SDL_GetKeyName(event.key.keysym.sym))); //add keydown to keys set
					break;
				case SDL_KEYUP:
					keys.erase(std::string(SDL_GetKeyName(event.key.keysym.sym))); //remove keyup from keys set
					break;
				case SDL_MOUSEMOTION:
					mouseX = event.motion.x;
					mouseY = event.motion.y;
					mouseDeltaX = event.motion.xrel;
					mouseDeltaY = event.motion.yrel;
					break;
				case SDL_MOUSEBUTTONDOWN:
					if (!buttons.contains(event.button.button)) {
						currentButtons.insert(event.button.button);
					}
					buttons.insert(event.button.button);
					break;
				case SDL_MOUSEBUTTONUP:
					buttons.erase(event.button.button);
					break;
				case SDL_MOUSEWHEEL:
					mouseScroll = event.wheel.y;
					break;
				}
			}

			cudaSetDevice(0);
			cudaMalloc(&d_pixels, static_cast<unsigned long long>(800) * 600 * sizeof(cell));
			cudaMemcpy(d_pixels, pixels, static_cast<unsigned long long>(600) * 800 * sizeof(cell), cudaMemcpyHostToDevice);
			
			if (mouseScroll != 0) {
				zoomFactor = pow(1.1, static_cast<float>(mouseScroll));
			}
			else if (keys.contains("Q")) {
				zoomFactor = pow(1.1, static_cast<float>(totalTime) / 60.0);
			}
			else if (keys.contains("E")) {
				zoomFactor = pow(1.1, -static_cast<float>(totalTime) / 60.0);
			}
			if (zoomFactor != 0.0) {
				localMouseX = topX + static_cast<float>(mouseX) / zoom;
				localMouseY = topY + static_cast<float>(mouseY) / zoom;
				topX = localMouseX + (topX - localMouseX) / zoomFactor;
				topY = localMouseY + (topY - localMouseY) / zoomFactor;
				zoom *= zoomFactor;
				zoomFactor = 0.0;
			}
			if (buttons.contains(1)) {
				topX -= static_cast<float>(mouseDeltaX) / zoom;
				topY -= static_cast<float>(mouseDeltaY) / zoom;
			}

			if (keys.contains("W")) {
				sImg += static_cast<float>(totalTime) / zoom / 60.0;
			}
			if (keys.contains("A")) {
				sReal -= static_cast<float>(totalTime) / zoom / 60.0;
			}
			if (keys.contains("S")) {
				sImg -= static_cast<float>(totalTime) / zoom / 60.0;
			}
			if (keys.contains("D")) {
				sReal += static_cast<float>(totalTime) / zoom / 60.0;
			}

			//Mandelbrot
			trueStart = now();
			if (timing) {
				calcTime = 0;
				drawTime = 0;
				start = now();
			}
			mandelbrot << <800, 600>> > (topX, topY, 1.0 / static_cast<float>(zoom), maxIter, d_pixels, sReal, sImg);
			cudaDeviceSynchronize();
			cudaMemcpy(pixels, d_pixels, 600 * 800 * sizeof(cell), cudaMemcpyDeviceToHost);
			if (timing) {
				calcTime += std::chrono::duration_cast<ms>(now() - start).count();
				start = now();
			}

			SDL_LockTexture(texture, NULL, &txtPixels, &pitch);
			Uint32* pixel_ptr = (Uint32*)txtPixels;
			for (uint16_t i = 0; i < 800; i++) {
				for (uint16_t j = 0; j < 600; j++) {
					pixel = pixels[j * 800 + i];
					if (pixel.iter == maxIter) {
						pixel_ptr[j * 800 + i] = SDL_MapRGB(format, 0, 0, 0);
					}
					else {
						color = pixel.iter + pixel.frac;
						pixel_ptr[j * 800 + i] = SDL_MapRGB(format, 255 * cos(color * 29.0), 255 * cos(color * 33.0), 255 * cos(color * 37.0));
					}
				}
			}
			SDL_UnlockTexture(texture);
			SDL_RenderCopy(renderer, texture, NULL, NULL);
			SDL_RenderPresent(renderer);
			if (timing) {
				drawTime += std::chrono::duration_cast<ms>(now() - start).count();
				std::cout << "Calc time: " << calcTime / 1000 << " Draw time: " << drawTime / 1000 << std::endl;
			}
			totalTime = std::chrono::duration_cast<ms>(now() - trueStart).count() / 1000;
			if (totalTime < 60) {
				SDL_Delay(60 - totalTime);
				totalTime = 60;
			}
			std::cout << "Total time: " <<  totalTime << std::endl;
			//cudaError_t err = cudaGetLastError();
			//std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
		}

		//Clean up
		cudaFree(d_pixels);
		SDL_FreeFormat(format);
		cudaDeviceReset();
		if (window) {
			SDL_DestroyWindow(window);
		}
		if (renderer) {
			SDL_DestroyRenderer(renderer);
		}
		TTF_Quit();
		Mix_Quit();
		IMG_Quit();
		SDL_Quit();
		return 0;
	}
	else {
		return 0;
	}
}