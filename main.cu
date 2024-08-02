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

const int WIDTH = 800, HEIGHT = 600;
typedef std::chrono::steady_clock::time_point tp;
typedef std::chrono::microseconds ms;
tp now() {
	return std::chrono::high_resolution_clock::now();
}
struct cell {
	uint8_t r, g, b;
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
cell pixels[WIDTH * HEIGHT];
cell* d_pixels;
int maxIter = 1000;
cell pixel;
bool timing = true;
const int FRAMECAP = 1000 / 30;
int totalTime = FRAMECAP;

void debug(int line, std::string file) {
	std::cout << "Line " << line << " in file " << file << ": " << SDL_GetError() << std::endl;
}

__global__ void mandelbrot(double topX, double topY, double delta, int max, cell pixels[WIDTH * HEIGHT], double sReal, double sImg) {
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
	if (i == max) {
		pixels[threadIdx.x * WIDTH + blockIdx.x] = { 0, 0, 0 };
	}
	else {
		double color = static_cast<float>(i) + log2(0.5 * log(img * img + real * real) / log(20.0));
		pixels[threadIdx.x * WIDTH + blockIdx.x] = { static_cast<unsigned char>(255.0 * cos(color * 29.0)), static_cast < unsigned char>(255.0 * cos(color * 33.0)), static_cast < unsigned char>(255.0 * cos(color * 37.0)) };
	}
}

int main(int argc, char* argv[]) {
	if (SDL_Init(SDL_INIT_EVERYTHING) == 0 && TTF_Init() == 0 && Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) == 0) {
		//Setup
		window = SDL_CreateWindow("Window", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIDTH, HEIGHT, 0);
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
			SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
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
			cudaMalloc(&d_pixels, static_cast<unsigned long long>(WIDTH) * HEIGHT * sizeof(cell));
			cudaMemcpy(d_pixels, pixels, static_cast<unsigned long long>(WIDTH) * HEIGHT * sizeof(cell), cudaMemcpyHostToDevice);
			
			if (mouseScroll != 0) {
				zoomFactor = pow(1.1, static_cast<float>(mouseScroll));
			}
			else if (keys.contains("Q")) {
				zoomFactor = pow(1.05, static_cast<float>(totalTime) / static_cast<float>(FRAMECAP));
			}
			else if (keys.contains("E")) {
				zoomFactor = pow(1.05, -static_cast<float>(totalTime) / static_cast<float>(FRAMECAP));
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
				topY -= 5.0 * static_cast<float>(totalTime) / zoom / static_cast<float>(FRAMECAP);
			}
			if (keys.contains("A")) {
				topX -= 5.0 * static_cast<float>(totalTime) / zoom / static_cast<float>(FRAMECAP);
			}
			if (keys.contains("S")) {
				topY += 5.0 * static_cast<float>(totalTime) / zoom / static_cast<float>(FRAMECAP);
			}
			if (keys.contains("D")) {
				topX += 5.0 * static_cast<float>(totalTime) / zoom / static_cast<float>(FRAMECAP);
			}

			//Mandelbrot
			trueStart = now();
			if (timing) {
				calcTime = 0;
				drawTime = 0;
				start = now();
			}
			mandelbrot << <WIDTH, HEIGHT>> > (topX, topY, 1.0 / static_cast<float>(zoom), maxIter, d_pixels, sReal, sImg);
			cudaDeviceSynchronize();
			cudaMemcpy(pixels, d_pixels, WIDTH * HEIGHT * sizeof(cell), cudaMemcpyDeviceToHost);
			if (timing) {
				calcTime += std::chrono::duration_cast<ms>(now() - start).count();
				start = now();
			}

			SDL_LockTexture(texture, NULL, &txtPixels, &pitch);
			Uint32* pixel_ptr = (Uint32*)txtPixels;
			for (uint16_t i = 0; i < WIDTH; i++) {
				for (uint16_t j = 0; j < HEIGHT; j++) {
					pixel = pixels[j * WIDTH + i];
					pixel_ptr[j * WIDTH + i] = SDL_MapRGB(format, pixel.r, pixel.g, pixel.b);
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
			if (totalTime < FRAMECAP) {
				SDL_Delay(FRAMECAP - totalTime);
				totalTime = FRAMECAP;
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