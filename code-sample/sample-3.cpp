#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int N = 10000; 
        Kokkos::View<float**, Kokkos::LayoutLeft>   A_LL("A_LL", N, N);      
        Kokkos::View<float**, Kokkos::LayoutRight>  A_LR("A_LR", N, N);       
        Kokkos::View<float*> row_sum("row_sum", N); 

        Kokkos::deep_copy(A_LL, 1);
        Kokkos::deep_copy(A_LR, 1);

        Kokkos::parallel_for("RowSumLL", N, KOKKOS_LAMBDA(int i) {
            float sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A_LL(i,j);
            }
            row_sum(i) = sum;
        });

        Kokkos::parallel_for("RowSumLR", N, KOKKOS_LAMBDA(int i) {
            float sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A_LR(i,j);
            }
            row_sum(i) = sum;
        });
    }
    Kokkos::finalize();
}