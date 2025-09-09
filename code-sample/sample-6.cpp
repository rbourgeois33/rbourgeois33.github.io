#include <Kokkos_Core.hpp>

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  { 
    const int size = 1<<15;
    const float fifth = 1.0/5;

    Kokkos::View<float**> A("A", size, size);
    Kokkos::View<float**> B("B", size, size);

    Kokkos::deep_copy(A, 0);
    Kokkos::deep_copy(A, 1);

    auto policy =  Kokkos::MDRangePolicy<Kokkos::Rank<2>> ({1,1}, {size-1, size-1});
    Kokkos::parallel_for("blurr", policy, KOKKOS_LAMBDA(const int i, const int j) { 
      
      float tmp=0;
      
      tmp += B(i-1,j);
      tmp += B(i,j);
      tmp += B(i+1,j);
      tmp += B(i  ,j+1);
      tmp += B(i  ,j-1);

      A(i,j)=tmp*fifth;

    });
  }
  Kokkos::finalize();
}  