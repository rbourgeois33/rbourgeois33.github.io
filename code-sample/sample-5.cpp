#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    const int size = 1<<27;

    Kokkos::View<float*> A("A", size);
    Kokkos::deep_copy(A, 1);

    int bound=3;

    Kokkos::parallel_for("Kernel", size, KOKKOS_LAMBDA(const int i) { 
        float tmp[3];

        for (int k=0; k<bound; k++){
          tmp[k] += A(i+k);
        }
        A(i) = tmp[0]+tmp[1]+tmp[2];

    });

  }
  Kokkos::finalize();
}