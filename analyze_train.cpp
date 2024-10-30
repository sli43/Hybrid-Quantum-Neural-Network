#include<iostream>
#include<fstream>
#include<random>
#include<vector>
#include <cstdlib>  // For rand() and srand()
#include <ctime>    // For time()

void hybrid::analyze_train(){
    read_in_hyperparameter();
    int sample_index;
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    sample_index=std::rand()%Ntrain;
    std::cout<<"analyze sample "<<sample_index<<std::endl;

/*
    std::vector<std::vector<int>> slices;
    slices.resize(10,std::vector<int>(10,0));
    for(int i=0; i<10; ++i){
      int k=0;
      for(int j=0; j<Ntrain; ++j){
           if(y_train[j][i]==1){
              slices[i][k]=j;
	      k++;
	      if(k==10) {break;}
	   }
      }
    }
*/
    int k=0;
    for(int i=0; i<Ntrain; ++i){
       if(y_train[i][7]>-1){
           batch=1;
           std::vector<std::vector<std::vector<double>>> xbatch;
           xbatch.resize(batch,std::vector<std::vector<double>>(imag_rows,std::vector<double>(imag_cols,0.0)));
           std::vector<std::vector<double>> ybatch;
           ybatch.resize(batch,std::vector<double>(Numclass,0.0));
           for(int kx=0; kx<imag_rows; ++kx){
             for(int ky=0; ky<imag_cols; ++ky){
                xbatch[0][kx][ky]=train_data[i][kx][ky];
             }
           }

           for(int y=0; y<Numclass; ++y) ybatch[0][y]=y_train[i][y];

           double loss=make_forward(xbatch,ybatch);
           std::cout<<k<<"   "<<i<<"   loss = "<<loss<<std::endl;
	   save_analyze_result(k,i);	   
	   k++;
       }
    }

}
