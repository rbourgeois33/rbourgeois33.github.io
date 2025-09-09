#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {

    const int size = 1<<27;
    int dim = 3;
    Kokkos::View<float**> A("A", size, dim);
    Kokkos::View<float**> B("B", size, dim);

    Kokkos::deep_copy(A, 0);
    Kokkos::deep_copy(A, 1);

    Kokkos::parallel_for("Kernel", size, KOKKOS_LAMBDA(const int i) { 
        for (int k = 0; k < 10; k++){
            for (int dir = 0; dir < dim; dir++){
                for (int dir2 = 0; dir2 < dim; dir2++){
                    A(i,dir) += B(i,dir2);
                }
            }
        }
    });

  }
  Kokkos::finalize();
}