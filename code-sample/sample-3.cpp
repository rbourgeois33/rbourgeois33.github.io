#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        int NR = 1000000;
        int NC = 64; 
        Kokkos::View<float**, Kokkos::LayoutLeft>   A_LL("A_LL", NR, NC);      
        Kokkos::View<float**, Kokkos::LayoutRight>  A_LR("A_LR", NR, NC);       
        Kokkos::View<float*> row_sum("row_sum", NR); 

        Kokkos::deep_copy(A_LL, 1);
        Kokkos::deep_copy(A_LR, 1);

        Kokkos::parallel_for("RowSumLL", NR, KOKKOS_LAMBDA(int i) {
            float sum = 0.0;
            for (int j = 0; j < NC; j++) {
                sum += A_LL(i,j);
            }
            row_sum(i) = sum;
        });

        Kokkos::parallel_for("RowSumLR", NR, KOKKOS_LAMBDA(int i) {
            float sum = 0.0;
            for (int j = 0; j < NC; j++) {
                sum += A_LR(i,j);
            }
            row_sum(i) = sum;
        });
    }
    Kokkos::finalize();
}