#include<iostream>
#include<fstream>
#include <math.h>
#include<complex>
#include<cmath>
#include<vector>
using namespace std;
void hybrid::read_in_training_data(){
     train_data.resize(Ntrain,vector<vector<double>>(imag_rows,vector<double>(imag_cols,0.0)));
     y_train.resize(Ntrain,vector<double>(10,0.0));
     test_data.resize(Ntest,vector<vector<double>>(imag_rows,vector<double>(imag_cols,0.0)));
     y_test.resize(Ntest,vector<double>(10,0.0));

     ifstream file1;
     int index;
     int k;
     file1.open("train.dat");
    if (!file1.is_open()) {
        std::cerr << "Unable to open train file!" << std::endl;
        std::cerr<<"   "<<std::endl;
        file1.close();
    }
     for(int i=0; i<Ntrain; ++i){
        file1 >> index;
        for (int ix=0; ix<imag_rows; ++ix){
           for(int iy=0; iy<imag_cols; ++iy){
              file1>>train_data[i][ix][iy];
           }
        }
        file1>>k;
        y_train[i][k]=1.0;
        //std::cout<<k<<"   "<<y_train[i][k]<<std::endl;
     }
     file1.close();
     
     ifstream file2;
     file2.open("test.dat");
    if (!file2.is_open()) {
        std::cerr << "Unable to open test file!" << std::endl;
        std::cerr<<"   "<<std::endl;
        file2.close();
    }
     for(int i=0; i<Ntest; ++i){
        file2 >> index;
        for (int ix=0; ix<imag_rows; ++ix){
           for(int iy=0; iy<imag_cols; ++iy){
              file2>>test_data[i][ix][iy];
           }
        }
        file2>>k;
        y_test[i][k]=1.0;
     }
     file2.close();
     
     
}




void hybrid:: read_in_hyperparameter(){
     std::string file_name;
     if(CNNmode=="C"){
        file_name="Cweight_"+std::to_string(jobid)+".dat";
     }else if(CNNmode=="H"){ 
        file_name="Hweight_"+std::to_string(jobid)+".dat";
     }
     ifstream myfile;
     myfile.open(file_name);
    // Check if the file is open
    if (!myfile.is_open()) {
        std::cerr << "Unable to open weight file!" << std::endl;
        std::cerr<<"   "<<std::endl;
	myfile.close();
    }

     int l1, l2, l3, l4;
     std::string str1, str2;

     std::getline(myfile,str1);
     myfile>>l1>>l2;
     for(int i=0; i<l1; ++i){
       for(int j=0; j<l2; ++j){
          myfile>>dense1_weight[i][j];
       }
       std::getline(myfile,str2);
     }

     std::getline(myfile,str1);
     myfile>>l1;
     for(int i=0; i<l1; ++i){
         myfile>>dense1_biase[i];
     }
     std::getline(myfile,str1);

     std::getline(myfile,str1);
     myfile>>l2>>l1;

     for(int i=0; i<l1; ++i){
       for(int j=0; j<l2; ++j){
          myfile>>dense2_weight[i][j];
       }
       std::getline(myfile,str2);
     }

     std::getline(myfile,str1);

     myfile>>l1;
     for(int i=0; i<l1; ++i){
         myfile>>dense2_biase[i];
     }


     myfile>>str2;

     myfile>>l1>>l2;
 
     for(int i=0; i<l1; ++i){
        for(int j=0; j<l2; ++j){
            myfile>>weightcnn[i][j];
	}
     }

     myfile>>str2;
     myfile>>l1;
     for(int i=0; i<l1; ++i){
          myfile>>biasecnn[i];
     }

     myfile>>str2;
     myfile>>l1;
     for(int i=0; i<l1; ++i){
         myfile>>moment[i]>>velocity[i];
     }

     myfile.close();
     pack_weight();
}
