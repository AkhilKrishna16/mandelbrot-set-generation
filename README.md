# Mandelbrot Set Generation (CPU vs CUDA GPU)

This project explores how GPU parallelism (via **CUDA**) accelerates Mandelbrot set generation compared to traditional CPU-based methods like **Pthreads** and **OpenMP**. I originally built a multithreaded CPU renderer, then rewrote the core computation in CUDA to compare real-world performance.

By offloading massively parallel per-pixel computations to the GPU, the CUDA version demonstrates why modern AI and HPC workloads rely so heavily on GPUs: the ability to run **tens of thousands of lightweight threads with almost no scheduling overhead** fundamentally changes what’s possible.

---

## 🚀 Final Performance Results

| Method             | Time (s)     |
|--------------------|--------------|
| **CUDA (GPU)**     | **0.125352** |
| **OpenMP (CPU)**   | **0.220818** |
| **Pthreads (CPU)** | **0.642524** |

- **CUDA is ~1.8× faster than optimized OpenMP**
- **CUDA is ~5× faster than Pthreads**

Even on a modest NVIDIA T4, the GPU performance jump is clear.

---

## 🧠 Why the Speedup?

The Mandelbrot algorithm is “embarrassingly parallel” — every pixel can be computed independently.  
GPUs excel here because they can schedule **tens of thousands (even millions)** of threads simultaneously with tiny context-switching costs.

- CPU threads are heavyweight  
- GPU threads are extremely lightweight  

This is why GPUs dominate:
- deep learning  
- scientific compute  
- physics simulations  
- graphics  
- HPC workloads  

---

## 🛠️ Build & Run Instructions

### Requirements
- NVIDIA GPU with CUDA support  
- CUDA Toolkit installed  
- GCC / Clang for CPU versions  
- `stb_image_write.h` included for PNG output  

---

### Build (GPU Version)

```bash
nvcc mandelbrot.cu -o mandelbrot_gpu
./mandelbrot_gpu

