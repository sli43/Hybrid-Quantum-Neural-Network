HDF5=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/hdf5-1.10.6-b4voav5jspfwo4qhsrqzxmem3zfourlz
HDF5_include=$(HDF5)/include
HDF5_library=$(HDF5)/lib -lhdf5 -lhdf5_cpp

qulacs_home=/home/shaozhl/library/qulacs
qulacs_include=$(qulacs_home)/include
qulacs_library=$(qulacs_home)/lib -lvqcsim_static -lcppsim_static -lcsim_static

BOOST_HOME=/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/boost-1.79.0-hship335kgdgemh2chsvvd7hswxo6vue
BOOST_INCLUDE=$(BOOST_HOME)/include
BOOST_LIBRARY=$(BOOST_HOME)/lib -lboost_program_options

All:
	mpicxx -g -std=c++17 -O3 -I$(HDF5_include) -I$(BOOST_INCLUDE) -I$(qulacs_include) hybrid.cpp -o hybrid -L$(qulacs_library) -L$(HDF5_library) -L$(BOOST_LIBRARY) -fopenmp 

old:
	c++ -pedantic -std=c++17 -Wall -Wextra -Weffc++ -O3 -DNDEBUG -DEIGN_NO_DEBUG -isystem /home/sli43/Documents/software/library/include/eigen3 -I/usr/local/include/qpp -I/usr/local/include/qpp/qasmtools/include  -I$(BOOST_INCLUDE) hybrid.cpp -o hybrid -L$(BOOST_LIBRARY)
