#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {

    const int size = 1<<27;

    // Create a 1D view of length 10
    Kokkos::View<float*> A("A", size);
    Kokkos::View<float*> B("B", size);

    // Initialize with zeros
    Kokkos::deep_copy(A, 0);
    Kokkos::deep_copy(A, 1);

    Kokkos::parallel_for("Kernel", size, KOKKOS_LAMBDA(const int i) { 

      float tmp=0;
      for (int k=0; k<10; k++){
        tmp += B(i-1);
        tmp += B(i);
        tmp += B(i+1);
      }

      A(i)+=tmp;
    });

  }
  Kokkos::finalize();
}