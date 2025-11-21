#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <time.h>

#define MAX_ITR      1000
#define IMG_W        1920
#define IMG_H        1080
#define CHANNELS     3

#define BLOCK_W      32
#define BLOCK_H      32

typedef struct complexNumber {
    double real;
    double imag;
} C;

__device__ void addComplex(const C *a, const C *b, C *out) {
    out->real = a->real + b->real;
    out->imag = a->imag + b->imag;
}

__device__ void mulComplex(const C *a, const C *b, C *out) {
    out->real = (a->real * b->real) - (a->imag * b->imag);
    out->imag = (a->real * b->imag) + (a->imag * b->real);
}

__device__ double absComplex(const C *z) {
    return sqrt(z->real * z->real + z->imag * z->imag);
}

__device__ float computeEscapeTime(const C *c) {
    C z = {0.0, 0.0};
    C z2;
    int i;

    for (i = 0; i < MAX_ITR; i++) {
        if (z.real * z.real + z.imag * z.imag > 4.0f)
            break;

        mulComplex(&z, &z, &z2);
        addComplex(&z2, c, &z);
    }

    if (i == MAX_ITR)
        return MAX_ITR;

    double mag = absComplex(&z);
    return i + 1 - log2f(log2f(mag));
}

__device__ void mapColor(
    float smooth,
    unsigned char *r,
    unsigned char *g,
    unsigned char *b
) {
    float t = smooth / (float)MAX_ITR;

    *r = (unsigned char)(9   * (1 - t) * t * t * t * 255);
    *g = (unsigned char)(15  * (1 - t) * (1 - t) * t * t * 255);
    *b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
}

__global__ void mandelbrotKernel(
    unsigned char *img,
    double realMin,
    double imagMin,
    double dReal,
    double dImag
) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= IMG_W || y >= IMG_H)
        return;

    double real = realMin + x * dReal;
    double imag = imagMin + y * dImag;

    C point = {real, imag};
    float smooth = computeEscapeTime(&point);

    unsigned char r, g, b;
    mapColor(smooth, &r, &g, &b);

    int idx = (y * IMG_W + x) * CHANNELS;
    img[idx]     = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

int main() {
    const double REAL_MIN = -2.0;
    const double REAL_MAX =  1.0;
    const double IMAG_MIN = -0.85;
    const double IMAG_MAX =  0.8375;

    double dReal = (REAL_MAX - REAL_MIN) / IMG_W;
    double dImag = (IMAG_MAX - IMAG_MIN) / IMG_H;

    size_t imgBytes = IMG_W * IMG_H * CHANNELS;

    unsigned char *hostImg = (unsigned char *)malloc(imgBytes);
    if (!hostImg) {
        fprintf(stderr, "Failed to allocate host image buffer.\n");
        return 1;
    }

    unsigned char *devImg;
    cudaMalloc(&devImg, imgBytes);

    dim3 blockDim(BLOCK_W, BLOCK_H);
    dim3 gridDim(
        (IMG_W + BLOCK_W - 1) / BLOCK_W,
        (IMG_H + BLOCK_H - 1) / BLOCK_H
    );

    clock_t start = clock();
    mandelbrotKernel<<<gridDim, blockDim>>>(devImg, REAL_MIN, IMAG_MIN, dReal, dImag);
    cudaDeviceSynchronize();
    clock_t end = clock();

    printf("Time taken: %.6f seconds\n",
           (double)(end - start) / CLOCKS_PER_SEC);

    cudaMemcpy(hostImg, devImg, imgBytes, cudaMemcpyDeviceToHost);

    if (!stbi_write_png("mandelbrot_set.png", IMG_W, IMG_H, CHANNELS,
                        hostImg, IMG_W * CHANNELS)) {
        fprintf(stderr, "Failed to output PNG file.\n");
        return 1;
    }

    cudaFree(devImg);
    free(hostImg);

    printf("Render saved as mandelbrot_set.png\n");
    return 0;
}
