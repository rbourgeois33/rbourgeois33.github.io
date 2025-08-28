# Basic performance tricks for porting kernels to the GPU with Kokkos

## Some context and motivations
Hello world! This is my first blog post. I'm Rémi Bourgeois, PhD. I am a researcher engineer working at the French Atomic Energy Comission (CEA). I work on the [TRUST platform](https://cea-trust-platform.github.io/), a HPC (multi-GPU) CFD code that serves as a basis for many research/industrial applications.

 I was hired by CEA to join the porting/optimization of this legacy code to the GPU using [Kokkos](https://github.com/kokkos/kokkos). This is quite a challenging task as the code is 20 years old, and more than 1400 kernels were identified to be ported to the GPU ! 
 
The nature of the task of porting code to the GPU, especially when time is limited, often lead to small mistakes that can undermine performance. The goal of this blogpost is to give you basic, easy tips to keep in mind when writing/porting your kernels, so that you get a *reasonable* performance. By *reasonable* I do not mean that you are getting *optimal* perfomance. In fact, I will not go over what I consider to be *advanced* optimization tricks such as the use of shared memory [link] or vectorized operations [link]. By *advanced*, I do not mean that these topics are especially difficult, but only that they require a significant design effort to be used effectively. In contrast, I believe that the tricks I will give to you in this blogpost are easy enought so that you can apply them while porting your code to the GPU in a time limited environement.
