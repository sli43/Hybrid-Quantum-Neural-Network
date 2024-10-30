#include "mpi.h"
#include<iostream>
#include<fstream>
#include<random>
#include<vector>

void hybrid::test_model(){
	if(choice==3) read_in_hyperparameter();
        std::ofstream test_out_file(test_out_file_name);
        // Check if the file is opened successfully
        if (!test_out_file.is_open()) {
               std::cerr << "Error opening test file!" << std::endl;
        }

        std::vector<std::vector<std::vector<double>>> xbatch;
        xbatch.resize(batch,std::vector<std::vector<double>>(imag_rows,std::vector<double>(imag_cols,0.0)));
        std::vector<std::vector<double>> ybatch;
        ybatch.resize(batch,std::vector<double>(Numclass,0.0));
        double loss;
        double accuracy; 
        double total_loss=0.0;
	int total_correct_number=0;
        correct_number=0;
	for(int p=0; p<Ntest/batch; p++){
            if(myrank==0) {std::cout<<"ith batch = "<<p<<"/"<<Ntest/batch<<std::endl;}
            for(int k=0; k<batch; k++){
                for(int kx=0; kx<imag_rows; kx++){
                    for(int ky=0; ky<imag_cols; ky++){
                        xbatch[k][kx][ky]=test_data[p*batch+k][kx][ky];
                    }
                 }
             for(int y=0; y<Numclass; y++) ybatch[k][y]=y_test[p*batch+k][y];
             }
               loss = make_forward(xbatch, ybatch);
               if(myrank==0){
                  total_loss+=loss;
                  total_correct_number+=correct_number;
                  std::cout<<"test loss = "<<loss/float(batch)<<"    accuracy =  "<<correct_number/float(batch)<<std::endl;
               }
               MPI_Barrier(MPI_COMM_WORLD);
               if(myrank==0){
		    test_out_file<<p<<"    ";
                    for(int i=0; i<layer5_rows; ++i){
                         test_out_file<<layer5_mid[0][i]<<"     ";
		    }
		    for(int i=0; i<layer5_rows; ++i){
                         test_out_file<<y_test[p*batch][i]<<"     ";
		    }
		    test_out_file<<loss/float(batch)<<"     "<<std::endl;
	       } 
	}
	if(myrank==0){
              accuracy=total_correct_number/float(Ntest);
              std::cout<<"total loss = "<<total_loss/float(Ntest)<<"      total accuracy = "<<total_correct_number/float(Ntest)<<std::endl;
              test_out_file.close();
	}

}
