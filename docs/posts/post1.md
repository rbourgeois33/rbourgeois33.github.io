# Six basic performance advices for porting kernels to the GPU.

## 0. Introduction
### Some context and motivations

Hello world ! This is my first blog post. I'm Rémi Bourgeois, PhD. I am a researcher engineer working at the French Atomic Energy Comission (CEA). I work on the [TRUST platform](https://cea-trust-platform.github.io/), a HPC (multi-GPU) CFD code that serves as a basis for many research/industrial applications.

I was hired by CEA to join the porting effort of this legacy code to the GPU using [Kokkos](https://github.com/kokkos/kokkos). This is quite a challenging task as the code is 20 years old, and more than 1400 kernels were identified to be ported to the GPU ! As I went and optimized some kernels, something struck me:

**The nature of the task of porting code to the GPU, especially when time is limited, often lead to small mistakes that can undermine performance.**

The goal of this blogpost is to give you *basic*, easy tips to keep in mind when writing / porting / first optimizing your kernels, so that you get a *reasonable* performance.

By applying them, I was able to get the following speedups that are measured relative to an already GPU-enabled baseline:

- A 40-50% speedup on a CFD [convection kernel](https://github.com/cea-trust-platform/trust-code/blob/509d09ae94bc5189131c6f160f1d42f6024cfa98/src/VEF/Operateurs/Op_Conv/Op_Conv_VEF_Face.cpp#L473) from TRUST (obtained on RTX A5000, RTX A6000 Ada and H100 GPUs). **Brace yourself**: this is a monstruous kernel.
- A 20-50% speedup on a CFD [diffusion kernel](https://github.com/cea-trust-platform/trust-code/blob/509d09ae94bc5189131c6f160f1d42f6024cfa98/src/VEF/Operateurs/Op_Diff_Dift/Op_Dift_VEF_Face_Gen.tpp#L192) from TRUST (obtained on RTX A6000 Ada and H100 GPUs).
- A 20% speedup on a [MUSCL reconstruction kernel](https://github.com/Maison-de-la-Simulation/heraclespp/blob/54feb467f046cf21bdca5cfa679b453961ea8d7e/src/hydro/limited_linear_reconstruction.hpp#L54) from the radiative hydrodynamics code [heraclescpp](https://github.com/Maison-de-la-Simulation/heraclespp) (obtained on a A100 GPU)
- TODO: add ncu reports
  
By *reasonable* I do not mean that you will get *optimal* performance. In fact, I will not go over what I consider to be *advanced* optimization advices such as the use of 

- [shared memory](https://www.youtube.com/watch?v=A1EkI5t_CJI&t=5s), 
- [vectorized operations](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/),
- [tensor cores operations](https://developer.nvidia.com/blog/optimizing-gpu-performance-tensor-cores/),
- [hardware-specific optimizations](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/?playlistId=playList-600dacf3-7db9-45fe-b0a2-e0156a792bc5). 
  
If getting  *optimal* performance is crucial to your application, consider learning more and apply these, but keep in mind that **performance often comes at the cost of portability**.
The advices are general enough so that they should allow speedups on all cards from all vendors.  By *advanced*, I do not mean that these topics are especially difficult or out of reach, but only that they require a significant design effort to be used effectively in a production context such as a wide CFD code like TRUST. In contrast, I believe that the advices I will give to you in this blogpost are easy enough so that you can, and should apply them straightforwardly while porting your code to the GPU in a time limited environement. 

**Note 1:** The target audience is engineer / researcher that want to get started with GPU porting in a code that relies on custom, domain specific low-level kernel. But do not reinvent the wheel ! i.e. do not rewrite kernels that have been implemented, hihgly optimized and distributed in libraries. Consider looking into (non exhaustive list !):

- [CUDA Libraries](https://docs.nvidia.com/cuda-libraries/index.html).
- [kokkos kernels](https://github.com/kokkos/kokkos-kernels) for portable BLAS, sparse BLAS and fraph kernels.
- [Trilinos](https://trilinos.github.io/) for high level, portable solutions for the solution of large-scale, complex multi-physics engineering and scientific problems.
- [PETSc](https://petsc.org/release/) for the scalable solution of scientific applications modeled by partial differential equations (PDEs)

### Disclaimers

If you think I wrote something that is wrong, or misleading please let me know !

I am running my performance tests on Nvidia GPUs, just because they are more easily available to me, and that I am more familiar with the performance tools such as [nsight systems](https://developer.nvidia.com/nsight-systems) (nsys) and [nsight compute](https://developer.nvidia.com/nsight-compute) (ncu). However, note that AMD provides similar profilers, and that the advices that I give here are simple enought so that they apply for GPUs from both vendors.

Moreover, I will use Kokkos as the programming model, just because I work with it, and that performance portability is **cool**. Again, the concepts are simple enought so that you can translate them to your favorite programming model, OpenMP, SYCL, Cuda, Hip.

### Pre-requisits

In this small tutorial, I will assume that you are already familiar with / will not cover:

- Basic C++.
- The reason why you might want to use the GPU, and that you need a big enough problem to make full use of it.
- How to Compile a GPU code, generate a report with [Nvidia nsight compute](https://youtu.be/04dJ-aePYpE?si=wTO9vJsRmVMBfM8a) and loading in with the ui.
- What is Kokkos, why you might want to use it and how to get started with it. Some ressources:
    - [Talk by Christian Trott, Co-leader of the Kokkos core team](https://www.youtube.com/watch?v=y3HHBl4kV7g).
    - [Kokkos lecture series](https://www.youtube.com/watch?v=rUIcWtFU5qM&list=PLqtSvL1MDrdFgDYpITs7aQAH9vkrs6TOF) (kind of outdated, but you can find a lot of ressources online, alos, join the slack !).
    -  **Note:** you really *should* consider using Kokkos, or any other portable programming model. It's good enough so that CEA adopted it for it's legacy codes ! (see [the CExA project](https://cexa-project.org/)).
- Basic GPU architecture, in particular:
    - That you should avoid host to device memory transfers.
    - The roofline performance model.
    - What does compute bound / memory bound mean.
    - Some knowledge about the memory hierarchy (registers, L1/L2 caches, DRAM) and the increasing cost of memory accesses.
    - Some ressources:
        - [1h30 lecture from Athena Elfarou on GPU architecture / CUDA programming](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/)
        - [13 lectures on CUDA programming by Bob Crovella](https://www.youtube.com/watch?v=OsK8YFHTtNs&list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj)
- Application-level optimization:
    - How to build a sensible optimization roadmap with e.g. Nvidia Nsight System
    - How to ensure that it is worth it to optimize the kernel you are looking (Don't assume bottleneck, profile, assess, optimize).
    - Some ressouces:
        -  [8th lecture from the Bob Crovella lecture series](https://www.youtube.com/watch?v=nhTjq0P9uc8&list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj&index=8) which focuses on that topic.

### Outline

The outline for this post is the following 6 rules of thumbs,or advices, largely inspired by [the Nvidia Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html):

1. Minimise redundant global memory accesses.
2. Ensure memory access are coalesced.
3. Minimize redundant math operation, use cheap arithmetics.
4. Understanding occupancy.
5. Avoid the use of *Local memory*.
6. Avoid thread divergence.

For each topic, I will provide:

- An explanation of why you need to worry about it, "Background" subsections,
- How to detect that it is limiting your kernel performance using ncu , "Profiler diagnosis" subsections,
- How to fix / avoid the issue, "Advices" subsections.

Feel free to jump straight into your sections of interest. 

### Before we start

Before going into the 6 advices, I invite you to read [my post on the cost of communications](posts/post2.md) that is a good, unesseray long introduction for advices 1. and 2. I also strongly advise watching [this brilliant talk on communication-avoiding algorithms](https://www.youtube.com/watch?v=sY3bgirw--4).

## 1. Minimise redundant global memory accesses
### Background
### Profiler diagnosis
### Advices
use register variable + static array, might need to template

## 2. Ensure memory access are coalesced
### Background
### Profiler diagnosis
### Advices
Think about your data Layout Kokkos layout conspiracy The granularity of memory accesses: lost bytes


## 3. Minimize redundant math operation, use cheap arithmetics
### Background
### Profiler diagnosis
### Advices
FMA, / vs *, unroll loop for int computation might need to template Math can be a bottleneck  Do smarter math

## 4. Understanding occupancy
### Background
### Profiler diagnosis
### Advices
Hide latency The occupancy trap, ILP, hide latency reduce Register usage
Reduce usage, launch bound, no MDRANGE, we dont talk about block and share memory limits, template away expensive branches

## 5. Avoid the use of *Local memory*
### Background
### Profiler diagnosis
### Advices
 Local memory is SLOW
Why does it spills + ref a la precedente section
How to avoid stack usage
attention aux tableaux statiques

## 6. Avoid thread divergence
### Background
### Profiler diagnosis
### Advices
The SIMD pattern, masking templating

## Final advices

Participate to hackathons !

