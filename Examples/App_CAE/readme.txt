This project has some necessary dependencies:
-------------------------------------------------------------
1. metis.lib (debug/release mode)
(../Extern/CAESolver/metis)
2. contact_impact_parallel.lib  (debug/release mode) and the corresponding header files
(../Extern/CAESolver/CAE_h)

Not necessary but sometimes may be helpful
3. cublas.lib
(../Extern/CAESolver)


Turn off other cuda project Generate Relocatable Device Code
-------------------------------------------------------------
Project properties -> CUDA C/C++ -> Generate Relocatable Device Code -> false