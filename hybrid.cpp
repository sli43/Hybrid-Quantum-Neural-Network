#include "mpi.h"
#include<iostream>
#include<fstream>
#include<random>
#include<vector>
//#include<boost/program_options.hpp>

#include<cppsim/state.hpp>
#include<cppsim/circuit.hpp>
#include<cppsim/observable.hpp>


#include "activation.cpp"
#include "qcnn.h"
#include "read_in_data.cpp"
#include "initial.cpp"
#include "training.cpp"
#include "analyze_train.cpp"
#include "perform_testing.cpp"

#include "forward.cpp"
#include "backward.cpp"
#include "quantum_circuit.cpp"
#include "optimize.cpp"
#include "io.cpp"

using namespace std;

void read_input_set(const char* inputname,int &fjobid,int &foption, int &fNtrain,int &ftest){
    std::ifstream file(inputname);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << inputname << std::endl;
    }

    file>>fjobid;
    file>>foption;
    file>>fNtrain;
    file>>ftest;

}


hybrid::hybrid(int fjobid, int foption, int fNtrain, int fNtest){
	jobid=fjobid;
	Ntrain=fNtrain;
	Ntest=fNtest;
        int ierr, myrank, nproc;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	read_in_training_data();
        
	initial();
        choice=foption;

	if(choice==1){
           train_model();
           test_model();
	}else if(choice==2){
           analyze_train();
	}else if(choice==3){
           test_model();
	}
}


int main(int argc, char* argv[]){
    
     int ierr, myrank, nproc;
     MPI_Status status;
     ierr=MPI_Init(&argc,&argv);
     if ( ierr != 0 ){
         std::cout << "  MPI_Init returned ierr = " << ierr << std::endl;
         exit ( 1 );
     }
     // get the number of processes
     MPI_Comm_size(MPI_COMM_WORLD, &nproc);
     // get the rank number
     MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
   //  std::cout<<myrank<<"   "<<nproc<<std::endl;

    if(myrank==0) std::cout<<"total number processors = "<<nproc<<std::endl;
    char* inputname = argv[1];
    int fjobid, fNtrain, fNtest, foption;
    
    read_input_set(inputname,fjobid, foption, fNtrain, fNtest);
    hybrid network(fjobid, foption, fNtrain, fNtest);
   
    MPI_Finalize();

}
