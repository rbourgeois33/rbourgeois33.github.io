#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int N = 10000; 
        Kokkos::View<float**, Kokkos::LayoutLeft>  A_LL("A_LL", N, N);       
        Kokkos::View<float*> row_sum("row_sum", N); 
        Kokkos::View<float*> col_sum("col_sum", N); 

        Kokkos::deep_copy(A_LL, 1);

        Kokkos::parallel_for("RowSumLL", N, KOKKOS_LAMBDA(int i) {
            float sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += A_LL(i,j);
            }
            row_sum(i) = sum;
        });
        Kokkos::parallel_for("ColSumLL", N, KOKKOS_LAMBDA(int j) {
            float sum = 0.0;
            for (int i = 0; i < N; i++) {
                sum += A_LL(i,j);
            }
            col_sum(j) = sum;
        });
    }
    Kokkos::finalize();
}