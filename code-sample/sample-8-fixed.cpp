#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {

    const int size = 1<<27;

    Kokkos::View<float*> A("A", size);

    Kokkos::deep_copy(A, 1.0f);

    float result = 0;
    float dx = 1.0f/size;

    Kokkos::parallel_reduce("Kernel", size, KOKKOS_LAMBDA(const int i, float& lsum) { 
      
      float x = ((float) i)*dx;

      if ( i%64 < 32 ){
        lsum += x ;
      }else{
        lsum += Kokkos::cosh(x);
        lsum -= Kokkos::sinh(x);
      }

    }, result);

  }
  Kokkos::finalize();
}