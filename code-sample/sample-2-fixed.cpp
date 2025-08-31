#include <Kokkos_Core.hpp>

template<int dim>
void apply_kernel(Kokkos::View<float**> A,  Kokkos::View<float**> B, int size){

    Kokkos::parallel_for("Kernel", size, KOKKOS_LAMBDA(const int i) { 
        
        float Atmp[dim], Btmp[dim];

        for (int dir = 0; dir < dim; dir++){
            Atmp[dir] = A(i, dir);
            Btmp[dir] = B(i, dir);
        }

        for (int k = 0; k < 10; k++){
            for (int dir = 0; dir < dim; dir++){
                for (int dir2 = 0; dir2 < dim; dir2++){
                    Atmp[dir] += Btmp[dir];
                }
            }
        }

        for (int dir = 0; dir < dim; dir++){
            A(i,dir) = Atmp[dir];
        }
      
    });
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {

    const int size = 1<<27;

    int dim = 3;

    Kokkos::View<float**> A("A", size, dim);
    Kokkos::View<float**> B("B", size, dim);

    Kokkos::deep_copy(A, 0);
    Kokkos::deep_copy(A, 1);

    if (dim==1){
        apply_kernel<1>(A, B, size);
    } else if (dim==2){
        apply_kernel<2>(A, B, size);
    } else if (dim==3){
        apply_kernel<3>(A, B, size);
    } else{
        // Need more instantiations ! Fail or warning 
    }

  }
  Kokkos::finalize();
}