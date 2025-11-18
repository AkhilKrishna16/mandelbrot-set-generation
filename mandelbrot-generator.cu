#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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

__device__ void complexAdd(C *z, C *cnst, C *res) {
    res->real = z->real + cnst->real;
    res->imag = z->imag + cnst->imag;
}

__device__ void complexMultiply(C *x, C *y, C *res) {
    res->real = (x->real * y->real) - (x->imag * y->imag);
    res->imag = (x->real * y->imag) + (x->imag * y->real);
}

__device__ double complexAbsolute(C *c) {
    return sqrt((c->real * c->real) + (c->imag * c->imag));
}

__device__ float mandelbrot(C *cnst) {
    C z = {0.0, 0.0};
    C zSq;
    int i;

    for(i = 0; i < MAX_ITR; i++) {
        if(z.real * z.real + z.imag*z.imag > 4.0) {
            break;
        }

        complexMultiply(&z, &z, &zSq);
        complexAdd(&zSq, cnst, &z);
    }

    if(i == MAX_ITR) {
        return MAX_ITR;
    }
    
    double mag = sqrt(z.real * z.real + z.imag * z.imag);
    float smooth = i + 1 - log2f(log2f(mag));
    return smooth;
}

__device__ void getColor(float smooth, unsigned char *r, unsigned char *g, unsigned char *b)
{
    float t = smooth / (float)MAX_ITR;
    *r = (unsigned char)(9*(1-t)*t*t*t*255);
    *g = (unsigned char)(15*(1-t)*(1-t)*t*t*255);
    *b = (unsigned char)(8.5*(1-t)*(1-t)*(1-t)*t*255);
}


__global__ void parallelMandelbrot(unsigned char *dev_image, double REAL_MIN, double IMAG_MIN, double INC_REAL, double INC_IMAG) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < IMG_W && y < IMG_H) {
        double real = REAL_MIN + ((double)x * INC_REAL);
        double imag = IMAG_MIN + ((double)y * INC_IMAG);
        C pcNum = {real, imag};
        float smooth = mandelbrot(&pcNum);
        int pixelIdx = (y * IMG_W + x) * CHANNELS;
        unsigned char r, g, b;
        getColor(smooth, &r, &g, &b);
        dev_image[pixelIdx + 0] = r;    
        dev_image[pixelIdx + 1] = g;
        dev_image[pixelIdx + 2] = b;
    }
}

int main() {
    double REAL_MIN = -2.0;
    double REAL_MAX = 1.0;
    double IMAG_MIN = -0.85;
    double IMAG_MAX = 0.8375;

    double INC_REAL = (REAL_MAX - REAL_MIN) / IMG_W;
    double INC_IMAG = (IMAG_MAX - IMAG_MIN) / IMG_H;

    size_t img_bytes = IMG_W * IMG_H * CHANNELS * sizeof(unsigned char);
    unsigned char *host_image = (unsigned char *)malloc(img_bytes);
    if(host_image == NULL) {
        fprintf(stderr, "FAILED TO ALLOCATE MEMORY\n");
        return 1;
    }

    unsigned char *dev_image;
    cudaMalloc(&dev_image, img_bytes);
    dim3 block_dim(32, 32, 1);
    dim3 grid_dim(ceil((float)IMG_W / 32), ceil((float)IMG_H / 32), 1);
    parallelMandelbrot<<<grid_dim, block_dim>>>(dev_image, REAL_MIN, IMAG_MIN, INC_REAL, INC_IMAG);
    cudaDeviceSynchronize();
    cudaMemcpy(host_image, dev_image, IMG_W * IMG_H * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if(!stbi_write_png("mandelbrot_set.png", IMG_W, IMG_H, CHANNELS, host_image, IMG_W * CHANNELS)) {
        fprintf(stderr, "Failed to write PNG\n");
        return 1;
    }

    free(host_image);
    cudaFree(dev_image);
    printf("Image(s) written!\n");
}