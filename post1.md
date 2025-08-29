# Basic performance tricks for porting kernels to the GPU.

## Some context and motivations
Hello world! This is my first blog post. I'm Rémi Bourgeois, PhD. I am a researcher engineer working at the French Atomic Energy Comission (CEA). I work on the [TRUST platform](https://cea-trust-platform.github.io/), a HPC (multi-GPU) CFD code that serves as a basis for many research/industrial applications.

I was hired by CEA to join the porting effort of this legacy code to the GPU using [Kokkos](https://github.com/kokkos/kokkos). This is quite a challenging task as the code is 20 years old, and more than 1400 kernels were identified to be ported to the GPU ! As I went and optimized some kernels, something struck me:
 
**The nature of the task of porting code to the GPU, especially when time is limited, often lead to small mistakes that can undermine performance.** 

The goal of this blogpost is to give you *basic*, easy tips to keep in mind when writing / porting / first optimizing your kernels, so that you get a *reasonable* performance. 

By applying them, I was able to get:
- A 40-50% speedup on a CFD [convection kernel](https://github.com/cea-trust-platform/trust-code/blob/509d09ae94bc5189131c6f160f1d42f6024cfa98/src/VEF/Operateurs/Op_Conv/Op_Conv_VEF_Face.cpp#L473) from TRUST (obtained on RTX A5000, RTX A6000 Ada and H100 GPUs). **Brace yourself**: this is a monstruous kernel.
- A 40-50% speedup on a CFD [diffusion kernel](https://github.com/cea-trust-platform/trust-code/blob/509d09ae94bc5189131c6f160f1d42f6024cfa98/src/VEF/Operateurs/Op_Diff_Dift/Op_Dift_VEF_Face_Gen.tpp#L192) from TRUST (obtained on RTX A6000 Ada and H100 GPUs).
- A 20% speedup on a [MUSCL reconstruction kernel](https://github.com/Maison-de-la-Simulation/heraclespp/blob/54feb467f046cf21bdca5cfa679b453961ea8d7e/src/hydro/limited_linear_reconstruction.hpp#L54) from the radiative hydrodynamics code [heraclescpp](https://github.com/Maison-de-la-Simulation/heraclespp)
 - TODO: add ncu reports
 - A

By *reasonable* I do not mean that you are getting *optimal* perfomance. In fact, I will not go over what I consider to be *advanced* optimization tricks such as the use of [shared memory](https://www.youtube.com/watch?v=A1EkI5t_CJI&t=5s), [vectorized operations](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) orlaunch bound tuning [link]. By *advanced*, I do not mean that these topics are especially difficult or out of reach, but only that they require a significant design effort to be used effectively in a production context such as a CFD code like TRUST. In contrast, I believe that the tricks I will give to you in this blogpost are easy enough so that you can and should apply them straightforwardly while porting your code to the GPU in a time limited environement. Moreover, I will not go into GPU model-specific optimizations [link elfarou], the advices are basic enough so that they should beneif on all cards.

## Disclaimer & Requirements
### Disclaimers

If you think I wrote something that is wrong, please let me know !

I am running my performance tests on Nvidia GPUs, just because they are more easily available to me, and that I am more familiar with the performance tools such as [nsight systems](https://developer.nvidia.com/nsight-systems) (nsys) and [nsight compute](https://developer.nvidia.com/nsight-compute) (ncu). However, note that AMD provides similar profilers, and that the advices that I give here are simple enought so that they apply for GPUs from both vendors.

Moreover, I will use Kokkos as the programming model, just because I work with it, and that performance portability is **cool**. Again, the concepts are simple enought so that you can translate it to your favorite programming model, OpenMP, SYCL, Cuda, Hip.

### Requirements

In this small tutorial, I will assume that you are already familiar with:
- Basic C++.
- The reason why you might want to use the GPU, and that you need a big enough problem to make full use of it.
- What is Kokkos, why you want to use it and how to use it. Some ressources:
  - [Talk by Christian Trott, Co-leader of the Kokkos core team](https://www.youtube.com/watch?v=y3HHBl4kV7g). 
  - [Kokkos lecture series](https://www.youtube.com/watch?v=rUIcWtFU5qM&list=PLqtSvL1MDrdFgDYpITs7aQAH9vkrs6TOF) (kind of outdated, but you can find a lot of ressources online, alos, join the slack !).
  - **Note:** you really *should* consider using Kokkos, or any other portable programming model. It's good enough so that CEA adopted it for it's legacy codes ! (see [the CExA project](https://cexa-project.org/)).
- Basic GPU architecture, in particular:
  - That you should avoid host to device memory transfers.
  - The roofline performance model.
  - What does compute bound / memory bound mean.
  - Some knowledge about the memory hierarchy (registers, L1/L2 caches, DRAM) and the increasing cost of memory accesses.
  - Some ressources:
    - [1h30 lecture from Athena Elfarou on GPU architecture / CUDA programming](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/)
    -  [13 lectures on CUDA programming by Bob Crovella](https://www.youtube.com/watch?v=OsK8YFHTtNs&list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj)
 - Application-level optimization:
   - How to build a sensible optimization roadmap.
   - How to ensure that it is worth it to optimize the kernel you are looking (Don't assume bottleneck, profile, assess, optimize).
   - How to use Nvidia Nsight System
   - Some ressouces:
     - [8th lecture from the Bob Crovella lecture series](https://www.youtube.com/watch?v=nhTjq0P9uc8&list=PL6RdenZrxrw-zNX7uuGppWETdxt_JxdMj&index=8) which focuses on that topic.

## Outline 
1. Use the basic functionnalities of nsight compute.
   1. Lancer depuis un cluster, read en local, commande pour un kernel kokkos.
2. Minimuse redundant global memory accesses.
   1. Comm vs comp photo demmel
   2. James demmel stuff
   3. accumulateur
   4. tableaux statiques
   5. in ncu
3. Ensure memory access are coalesced.
   1. Sectors/cache line
   2. Layout (Kokkos Layout consipracy)
   3. in ncu
4. Minimize redundant math operation, use cheap arithmetics.
   1. FMA
   2. / vs *
   3. in ncu
5. Avoid the use of *Local memory* by removing:
   1. Register spilling
   2. accidental stack usage
      1. Accés statiques aux tableaux statiques. Cad avoir des bornes de boucles connues au compile time quand on accède à des tableaux statique dedans. Ca implique aussi des switch moches parfois quand on a le pattern y=face[x] avec x pas connu au compile time mais très très important pour la perf.
   3. in ncu, avec les options de compil
6. Understanding occupancy, and when to worry about it:
   1. latency hiding
   2. The occupancy trap, ILP
   3. Pas de MDRange: moins de pression sur les registres
7. Avoid thread divergence:
   1. Templater les paramètres qui change bcp l'execution. bien aussi pour l'occupancy !!


## Final advices
Participate to hackathons !