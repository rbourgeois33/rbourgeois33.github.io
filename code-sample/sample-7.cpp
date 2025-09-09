#include <Kokkos_Core.hpp>

const float fifth = 1.0/5;
const int size = 1<<15;

KOKKOS_INLINE_FUNCTION
void expensive_function(Kokkos::View<float**> B, float& tmp, const int i, const int j) {
  float v = B(i,j);

  // Inline array to hold many live values
  float arr[128];

  for (int k = 0; k < 128; ++k) {
    //runtime indexing so that the compiler cannot detect the reduce pattern,
    //fuse the two loops and only une one float register
    int ii = (i+j+k)%size;
    arr[k] = v+B(ii,j);
  }

  // Use arr only at the very end → forces them to stay live
  float sum = 0.0f;
  for (int k = 0; k < 128; ++k) {
    sum += arr[k];
  }

  tmp += sum;
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  { 

    Kokkos::View<float**> A("A", size, size);
    Kokkos::View<float**> B("B", size, size);

    Kokkos::deep_copy(A, 0);
    Kokkos::deep_copy(A, 1);

    bool expensive_option = false;

    Kokkos::parallel_for("blurr", (size-2)*(size-2), KOKKOS_LAMBDA(const int ilin) { 
      
      const int i = 1 + (ilin % (size-2));
      const int j = 1 + (ilin / (size-2));

      float tmp = 0.0f;

      tmp += B(i-1,j);
      tmp += B(i,j);
      tmp += B(i+1,j);
      tmp += B(i  ,j+1);
      tmp += B(i  ,j-1);
      tmp *= fifth;

      if (expensive_option) {
        expensive_function(B, tmp, i, j);
      }

      A(i,j) = tmp;
    });

  }
  Kokkos::finalize();
}