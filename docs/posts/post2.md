# The cost of memory transfers
## Disclaimer
This post was orginally supposed to be a short introduction within [my first post on GPU kernel optimisation](post1.md), but then I realized that I liked to talk too much about it and went out of scope. This is **largely** inspired by [this brilliant talk](https://www.youtube.com/watch?v=sY3bgirw--4) by Prof. James Demmel (Berkley), as well as a the [his CS267 class](https://sites.google.com/lbl.gov/cs267-spr2022) with free lectures on Youtube, where you can find **everything** that I explain here.

**Note** The terms *communications* and *memory transfers* will be used interchangeably. Also in the present context, a *word* refers to a single FP64 number.

## Hardware trends
Looking at Figure 1, memory transfers, either within DRAM, or over the network, have been more expensive than (FP64) math operation since ~1992:
![ ](image.png)
**Figure 1:** Evolution of the time per flop (gamma), inverse bandwitdh (beta) and latency (alpha) between ~1980 to ~2015. [Source](https://extremecomputingtraining.anl.gov/wp-content/uploads/sites/96/2025/08/Communication-Avoiding-Algorithms-for-Linear-Algebra-Machine-Learning-and-Beyond-v2_ATPESC-2025.pdf).

The graph stops around 2015, where the ratio of gamma to beta (DRAM) was around 10. Let's look at the current FP64 **Flop Per Load (FPL) factor** for more recent hardware:

| GPU   |   Release Year |   FP64 FLOPS (TFLOPS) |   BW (TB/s)  |   FPL |
|:------|---------------:|----------------------:|------------:|-----------------------:|
| [V100](https://www.techpowerup.com/gpu-specs/tesla-v100-pcie-16-gb.c2957)  |           2017 |                   7.066 |         0.897  |           ~65.19  |
| [A100](https://www.techpowerup.com/gpu-specs/a100-pcie-40-gb.c3623)  |           2020 |                   9.746 |         1.56     |           ~49.9     |
| [H100](https://www.techpowerup.com/gpu-specs/h100-pcie-80-gb.c3899)  |           2022 |                  25.61   |         2.04 |          ~ 100       |
| [B200](https://www.techpowerup.com/gpu-specs/b200-sxm-192-gb.c4210)  |           2024 |                  62   |         8.20      |           ~60       |   

**Table 1:** Evolution of the BW, FP64 flops and FPL for recent Nvidia GPU models. FPL is computed as $\frac{FP64 \ Flops}{BW}*8$ since a FP64 number is made of 8 bytes.

**Note** It is illuminating to think about FPL as the average amount of FP64 operation you can do in the time that it takes to load one word from DRAM.

As we can see, the FPL has been oscillating between 50 and 100. This is really large, and should motivate you to think really hard about how you access memory. In fact, *communication avoiding algorithms* have been a very active and fruitful research area which lead to the development of the BLAS standard, the Lapack library, and much more.

## A simple memory model

One result that I like a lot is the one presented in [the second CS267 lecture](https://www.youtube.com/watch?v=ictIJF2WXHE) where the following simple memory model is proposed:

- Assume a simple machine with just 2 levels of memory, fast and slow (think of e.g. DRAM / registers) and the following properties and notations:
    - $M=$ number of words that fits into fast memory,
    - no latency (simplifying assumption),
    - $t_m=$ time per slow memory operation e.g. to moove a word from fast to slow memory (inverse BW from Table 1 multiplied by 8 in our case since we are doing FP64 and ignoring latency),
    - $t_f=$ time per artithmetic operation i.e. the inverse of the FLOPS in Table 1.
- Assume an implementation of an algorithm with:
    - $m=$ number of words moved bewteen fast and slow memory to complete the algorithm,
    - $f=$ number of arithmetic operations to complete the algorithm,


We can then define $CI_{\text{implem}}=\frac{f}{m}$, a property  **of the implementation of the algorithm** that is called *the computational intensity*.It is the average number of flops per slow memory access. While the previously defined FPL factor, **a property of the machine**, is just given as $FPL_{\text{hardware}}=\frac{t_m}{t_f}$.

**Note:** Nvidia GPUs have 4 levels of memory: DRAM, L2 and L1 caches, and registers. Each level has ~an order of magnitude difference in bandwithd. CPUs have 5 levels with an additional L3 cache. Real memory models are super complicated ! However, it is clear that memory hierachies are omnipresent. As a result, the advices presented here also help CPU performance !

## Getting good performance

The minimum possible time for the our algorithm is $t_{\text{ideal}}=ft_f$, which is attained when the problem fits in fast memory ($m<M$) and no slow memory transaction are required. This implies that we don't read any intial data from slow memory nor store in it, this is never the case in practice. 

Let's compare this to the real time for a big enough problem $t_{\text{real}}=ft_f+mt_m$ which rewrites:

$t_{\text{real}}= t_{\text{ideal}}(1+\frac{FPL_{\text{hardware}}}{CI_{\text{implem}}})$

It is now clear that to get near optimal performance, we want to reduce the ratio $\frac{FPL_{\text{hardware}}}{CI_{\text{implem}}}$ as much as possible. Since ${FPL_{\text{hardware}}}$ is a property of the hardware, with a value ranging between 50 and 100 depending on the GPU considered, all we can do is try to reduce $\frac{1}{CI_{\text{implem}}}=\frac{m}{f}$, by trying to **reuse** the data we load as much as possible.

## Properties of an algorithm vs. properties of an implementation
I insist on using the terminology **implementation of an algorithm** because in practice, the numbers $f$, $m$ and $CI_{\text{implem}}=\frac{f}{m}$ should not be obtained by simply computing the ratio of how much memory should be touched, and how many operation should be done ideally, optimally for a given algorithm. Because most real problems do not fit in cache. Instead, these numbers are a property of how the algorithm is implemented, compiled and ran. In fact, they can vary dramatically between a naive and a smart implementation.

Let's consider the example of a very generic algorithm: **dense matrix multiplication**, $C=A.B$ (the 1st homework of [CS267](https://sites.google.com/lbl.gov/cs267-spr2022) and topic of the 2nd and 3rd lectures). If three $n\times n$ matrices fits in fast memory, we know that that we need to load/store only $3n^2$ words (2 matrix read, 1 matrix write) from slow memory, and perform $2n^3$ operations (one dot product per element of C, each dot product being $n$ multiply and $n$ add) with a resulting $CI_{\text{ideal}}^{\text{matmul}}=\frac{3}{2n}$. The bigger $n$ is, the closer we get from ideal performance. However, as $n$ grows, it is clear that the problem does not fit in fast memory ($3n^2>M$ eventually). Then, a naive implementation of matrix multiply such as:
```python
for i in range(n):
    #load A[i,:] in fast memory
    for j in range (n): 
        #load B[:,j] in fast memory
        C[i,j] = dot(A[i,:], B[:,j])
    #store C[i,:] in fast memory
```
can be shown to have a computational intensity $CI_{\text{naive}}^{\text{matmul}}=\mathcal{O}(1)$, which is terrible ! On the other hand, the well-known blocked implementation that splits and iterates over $b\times b$ sub-blocks of the matrices has a computational intensity of  $CI_{\text{blocked}}^{\text{matmul}}=\mathcal{O}(b)$, assuming that the blocks fit in fast memory. 

So, you might wonder, what should I do ? How do I know if there is a better algorithm ? Well, a theoretical upper bound on the computational intensity has been found and is given by $CI_{\text{blocked}}^{\text{matmul}}=\mathcal{O}(\sqrt{M})$, and if you ever write a new dense matmul implementation, you should strive to reach it. And notice ! the blocked algortihm reaches that bound. Indeed, since the blocks fit in fast memory, $3b^2 <M$ $\implies$ $b=\mathcal{O}(\sqrt{M})$. This is the whole point of *communication avoiding algorithms* research: computing lower bounds and finding algorithm that reaches them. Again, if you find this interesting, consider looking at this [brilliant introduction](https://www.youtube.com/watch?v=sY3bgirw--4).

## Conclusion

Well, all that is quite fascinating, but also overwhelming don't you think ? Well, you might not have to think about all this lower bound theory to get good speedups. In [my first post on GPU kernel optimisation](post1.md) I go over frequent coding mistakes that leads to extra useless communications. In that post, I will not give guidelines to reach theoritical lower bounds for your case. This is just too general to be discussed in a blogpost. As we saw, it constitutes a resarch topic on it's own and implies a deep re-thinking of the algorithms and data structures. No, here we will stay simple and focus on the following: given a GPU kernel, what frequent coding mistakes should we avoid to limit the amount of data we load/store from slow memory.