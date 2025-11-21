#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <time.h>

#define MAX_ITR 1000
#define NUM_THREADS 32 * 32
#define IMG_W 1920
#define IMG_H 1080
#define CHANNELS 3
#define NUM_BLOCKS (int)ceil(((double)IMG_W * IMG_H) / NUM_THREADS)

typedef struct complexNumber {
    double real;
    double imag;
} C;

__device__ void addComplex(C *a, C *b, C *out) {
    out->real = a->real + b->real;
    out->imag = a->imag + b->imag;
}

__device__ void mulComplex(C *a, C *b, C *out) {
    out->real = (a->real * b->real) - (a->imag * b->imag);
    out->imag = (a->real * b->imag) + (a->imag * b->real);
}

__device__ double absComplex(C *z) {
    return sqrt(z->real * z->real + z->imag * z->imag);
}

__device__ float computeEscapeTime(C *c) {
    C z = {0.0, 0.0};
    C z2;
    int i;

    for (i = 0; i < MAX_ITR; i++) {
        if (z.real * z.real + z.imag * z.imag > 4.0) break;

        mulComplex(&z, &z, &z2);
        addComplex(&z2, c, &z);
    }

    if (i == MAX_ITR) return MAX_ITR;

    double mag = absComplex(&z);
    float smooth = i + 1 - log2f(log2f(mag));
    return smooth;
}

__device__ void mapColor(float smooth, unsigned char *r, unsigned char *g, unsigned char *b) {
    float t = smooth / (float)MAX_ITR;
    *r = (unsigned char)(9  * (1-t) * t*t*t * 255);
    *g = (unsigned char)(15 * (1-t) * (1-t) * t*t * 255);
    *b = (unsigned char)(8.5*(1-t) * (1-t) * (1-t) * t * 255);
}

__global__ void mandelbrotKernel(
    unsigned char *img,
    double REAL_MIN, double IMAG_MIN,
    double dReal, double dImag
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < IMG_W && y < IMG_H) {
        double real = REAL_MIN + x * dReal;
        double imag = IMAG_MIN + y * dImag;

        C point = {real, imag};

        float val = computeEscapeTime(&point);

        int idx = (y * IMG_W + x) * CHANNELS;

        unsigned char r, g, b;
        mapColor(val, &r, &g, &b);

        img[idx    ] = r;
        img[idx + 1] = g;
        img[idx + 2] = b;
    }
}

int main() {
    double REAL_MIN = -2.0;
    double REAL_MAX = 1.0;
    double IMAG_MIN = -0.85;
    double IMAG_MAX = 0.8375;

    double dReal = (REAL_MAX - REAL_MIN) / IMG_W;
    double dImag = (IMAG_MAX - IMAG_MIN) / IMG_H;

    size_t img_bytes = IMG_W * IMG_H * CHANNELS;
    unsigned char *host_img = (unsigned char *)malloc(img_bytes);

    unsigned char *dev_img;
    cudaMalloc(&dev_img, img_bytes);

    dim3 blockDim(32, 32);
    dim3 gridDim(ceil((float)IMG_W / 32), ceil((float)IMG_H / 32));

    clock_t start = clock();
    mandelbrotKernel<<<gridDim, blockDim>>>(dev_img, REAL_MIN, IMAG_MIN, dReal, dImag);
    cudaDeviceSynchronize();
    clock_t end = clock();

    printf("Time taken: %f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

    cudaMemcpy(host_img, dev_img, img_bytes, cudaMemcpyDeviceToHost);

    if (!stbi_write_png("mandelbrot_set.png", IMG_W, IMG_H, CHANNELS, host_img, IMG_W * CHANNELS)) {
        fprintf(stderr, "Failed to write PNG\n");
    }

    free(host_img);
    cudaFree(dev_img);
    return 0;
}
