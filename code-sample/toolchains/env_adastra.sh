# Do not use, just for me. See https://github.com/rbourgeois33/rocprof_tests/tree/adastra-03-26
module purge
cd $WORK
module use rocm
module load python rocm-7.2.0
source venv_rocprof_compute/bin/activate
export HSA_XNACK=1
cd rbourgeois33.github.io/code-sample/