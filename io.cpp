#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<complex>
#include<H5Cpp.h>

void hybrid:: initial_history_output(){
     H5:: H5File history_file(history_file_name,H5F_ACC_TRUNC);

     H5::Group group1=history_file.createGroup("/gradient");
     H5::Group group2=history_file.createGroup("/weight");
     H5::Group group3=history_file.createGroup("/gradient_distribution");
     H5::Group group4=history_file.createGroup("/weight_distribution");
     H5::Group group5=history_file.createGroup("/parameter");
     H5::Group group6=history_file.createGroup("/moment");
     H5::Group group7=history_file.createGroup("/velocity");
     H5::Group group8=history_file.createGroup("/sample1");
     group1.close();
     group2.close();
     group3.close();
     group4.close();
     group5.close();
     group6.close();
     group7.close();
     group8.close();
     history_file.close();

}
void hybrid::save_parameter(){
     hsize_t dims[1];
     dims[0]=1;
     int Rank=1;
     H5::H5File file(history_file_name,H5F_ACC_RDWR);
     H5::DataSpace dataspace(Rank,dims);
     H5::Group group;
     group=file.openGroup("parameter");
     H5::DataSet dataset1=group.createDataSet("Ntrain",H5::PredType::NATIVE_INT,dataspace);
     dataset1.write(&Ntrain,H5::PredType::NATIVE_INT);
     dataset1.close();


     H5::DataSet dataset2=group.createDataSet("Ntest",H5::PredType::NATIVE_INT,dataspace);
     dataset2.write(&Ntest,H5::PredType::NATIVE_INT);
     dataset2.close();

     H5::DataSet dataset3=group.createDataSet("epoch",H5::PredType::NATIVE_INT,dataspace);
     dataset3.write(&epoch,H5::PredType::NATIVE_INT);
     dataset3.close();


     H5::DataSet dataset4=group.createDataSet("batch",H5::PredType::NATIVE_INT,dataspace);
     dataset4.write(&batch,H5::PredType::NATIVE_INT);
     dataset4.close();

 
     H5::DataSet dataset5=group.createDataSet("nproc",H5::PredType::NATIVE_INT,dataspace);
     dataset5.write(&nproc,H5::PredType::NATIVE_INT);
     dataset5.close();


     group.close();
     file.close();


}
void hybrid::save_history(int datatype, int time, const string & dataname,const int & N, const std::vector<double>& A){

     hsize_t dims[1];
     dims[0]=N;
     int Rank=1;
     H5::H5File file(history_file_name,H5F_ACC_RDWR);
     H5::DataSpace dataspace(Rank,dims);
     std::string secondstring = "_";
     std::string new_dataname=dataname + secondstring + std::to_string(time);
     H5::Group group;
     if(datatype==1) {
            group=file.openGroup("gradient");}
     else if(datatype==2){
	    group=file.openGroup("weight");
     }else if(datatype==3){
            group=file.openGroup("moment");
     }else if(datatype==4){
            group=file.openGroup("velocity");
     }
     H5::DataSet dataset=group.createDataSet(new_dataname,H5::PredType::NATIVE_DOUBLE,dataspace);
     dataset.write(A.data(),H5::PredType::NATIVE_DOUBLE);

     dataset.close();
     group.close();
     file.close();

}

void hybrid::save_history_sample(int time){

    hsize_t dims[1];
    dims[0]=layer1_rows*layer1_cols;
    int Rank=1;
    H5::DataSpace dataspace(Rank,dims);

    std::vector<double> data1;
    data1.resize(dims[0],0.0);
    std::string dataname;

    H5::H5File file(history_file_name,H5F_ACC_RDWR);
    H5::Group parentgroup;
    parentgroup=file.openGroup("sample1");
    std::string groupname;
    groupname="time"+std::to_string(time);
    H5::Group group=parentgroup.createGroup(groupname);

    for(int l=0; l<layer1_feature; ++l){
         dataname="output_feature_"+std::to_string(l);
         H5::DataSet dataset=group.createDataSet(dataname,H5::PredType::NATIVE_DOUBLE,dataspace);
         for(int i=0; i<layer1_rows*layer1_cols; ++i){
             data1[i]=sample_layer1[l][i];
	 }
	 dataset.write(data1.data(),H5::PredType::NATIVE_DOUBLE);
         dataset.close();	 
    }

    dims[0]=layer1_rows*layer1_cols*Nquant;
    H5::DataSpace dataspace1(Rank,dims); 
    data1.resize(dims[0],0.0);

    for(int l=0; l<layer1_feature; ++l){
        dataname="gradient_feature_"+std::to_string(l);
	H5::DataSet dataset=group.createDataSet(dataname,H5::PredType::NATIVE_DOUBLE,dataspace1);
        int ll=0;
	for(int i=0; i<layer1_rows*layer1_cols; ++i){
           for(int j=0; j<Nquant; ++j){
               data1[ll]=sample_gradient_layer1[l][i][j];
	       ll++;
	   }
	}
	 dataset.write(data1.data(),H5::PredType::NATIVE_DOUBLE);
         dataset.close();
    }

    group.close();
    parentgroup.close();
    file.close();


}

void hybrid::save_history_distribution(int datatype, int time, const string & dataname,const int & N, const std::vector<long int>& A){
     hsize_t dims[1];
     dims[0]=N;
     int Rank=1;
     H5::H5File file(history_file_name,H5F_ACC_RDWR);
     H5::DataSpace dataspace(Rank,dims);
     std::string secondstring = "_";
     std::string new_dataname=dataname + secondstring + std::to_string(time);
     H5::Group group;
     if(datatype==1) {
            group=file.openGroup("gradient_distribution");}
     else if(datatype==2){
            group=file.openGroup("weight_distribution");
     }
     H5::DataSet dataset=group.createDataSet(new_dataname,H5::PredType::NATIVE_LONG,dataspace);
     dataset.write(A.data(),H5::PredType::NATIVE_LONG);

     dataset.close();
     group.close();
     file.close();
}

void hybrid::save_analyze_result(const int & slice, const int & sample_index){
     std::string file_name="analyze_jobid_"+std::to_string(jobid)+".h5";
     H5:: H5File file;
     if(slice==0){
         file=H5::H5File(file_name,H5F_ACC_TRUNC);
     }else{
         file=H5::H5File(file_name,H5F_ACC_RDWR);
     }

     std::string sample_name="/sample_"+std::to_string(slice);
     H5::Group parentGroup=file.createGroup(sample_name);

     H5::Group group1=parentGroup.createGroup("input");
     H5::Group group2=parentGroup.createGroup("layer_cnn");
     H5::Group group3=parentGroup.createGroup("layer_pool");
     H5::Group group4=parentGroup.createGroup("layer_end");
     H5::Group group5=parentGroup.createGroup("output");


     hsize_t dims[1];

     dims[0]=imag_rows*imag_cols;
     H5::DataSpace dataspace(1,dims);
     std::vector<double> fx(imag_rows*imag_cols,0.0);
     int p=0;
     for(int kx=0; kx<imag_rows; ++kx){
       for(int ky=0; ky<imag_cols; ++ky){
           fx[p]=train_data[sample_index][kx][ky];
	   p++;
       }
     }
     std::string dname="input";
     H5:: DataSet dataset=group1.createDataSet(dname,H5::PredType::NATIVE_DOUBLE,dataspace);
     dataset.write(fx.data(),H5::PredType::NATIVE_DOUBLE);
     dataset.close();
     group1.close();

     dims[0]=layer1_rows*layer1_cols;
     H5::DataSpace dataspace2(1,dims);
     fx.resize(layer1_rows*layer1_cols);
     for(int i=0; i<layer1_feature; ++i){
         std::fill(fx.begin(),fx.end(),0.0);
	 int p=0;
	 for(int kx=0; kx<layer1_rows; ++kx){
	    for(int ky=0; ky<layer1_cols; ++ky){
                fx[p]=layer1[0][i][kx][ky];
		p++;
	    }
	 }
	 dname="feature_"+std::to_string(i);
         H5:: DataSet dataset=group2.createDataSet(dname,H5::PredType::NATIVE_DOUBLE,dataspace2);
	 dataset.write(fx.data(),H5::PredType::NATIVE_DOUBLE);
	 dataset.close();
     }

     group2.close();



     dims[0]=layer2_rows*layer2_cols;
     H5::DataSpace dataspace3(1,dims);
     fx.resize(layer2_rows*layer2_cols);
     for(int i=0; i<layer1_feature; ++i){
         std::fill(fx.begin(),fx.end(),0.0);
         int p=0;
         for(int kx=0; kx<layer2_rows; ++kx){
            for(int ky=0; ky<layer2_cols; ++ky){
                fx[p]=layer2[0][i][kx][ky];
                p++;
            }
         }
         dname="feature_"+std::to_string(i);
         H5:: DataSet dataset=group3.createDataSet(dname,H5::PredType::NATIVE_DOUBLE,dataspace3);
         dataset.write(fx.data(),H5::PredType::NATIVE_DOUBLE);
         dataset.close();
     }
     group3.close();


     dims[0]=layer5_rows;
     H5::DataSpace dataspace4(1,dims);
     fx.resize(layer5_rows);
     std::fill(fx.begin(),fx.end(),0.0);
     p=0;
     for(int i=0; i<layer5_rows; ++i){
        fx[i]=layer5[0][i];
     }
     dname="layer_end";
     dataset=group4.createDataSet(dname,H5::PredType::NATIVE_DOUBLE,dataspace4);
     dataset.write(fx.data(),H5::PredType::NATIVE_DOUBLE);
     dataset.close();
     group4.close();

     dims[0]=layer5_rows;
     H5::DataSpace dataspace5(1,dims);
     fx.resize(layer5_rows);
     std::fill(fx.begin(),fx.end(),0.0);
     p=0;
     for(int i=0; i<layer5_rows; ++i){
        fx[i]=y_train[sample_index][i];
     }
     dname="output";
     dataset=group5.createDataSet(dname,H5::PredType::NATIVE_DOUBLE,dataspace5);
     dataset.write(fx.data(),H5::PredType::NATIVE_DOUBLE);
     dataset.close();
     group5.close();
     parentGroup.close();
     file.close();


}


void hybrid:: save_data(){
   std::string file_name;
   ofstream myfile;
   if(CNNmode=="C"){
      file_name="Cweight_"+std::to_string(jobid)+".dat";
   }else if(CNNmode=="H"){
      file_name="Hweight_"+std::to_string(jobid)+".dat";
   }

   myfile.open(file_name);

   myfile<<"dense1_weight"<<std::endl;
   myfile<<layer4_rows<<"    "<<layer3_rows<<std::endl;
   for(int i=0; i<layer4_rows; ++i){
     for(int j=0; j<layer3_rows; ++j){
         myfile<<dense1_weight[i][j]<<"          ";
     }
     myfile<<std::endl;
   }
   
   
   myfile<<"dense1_biase"<<std::endl;
   myfile<<layer4_rows<<std::endl;
   for(int i=0;i<layer4_rows; ++i){
      myfile<< dense1_biase[i]<<std::endl;
   }


   myfile<<"dense2_weight"<<std::endl;
   myfile<<layer4_rows<<"    "<<layer5_rows<<std::endl;
   for(int i=0; i<layer5_rows; ++i){
     for(int j=0; j<layer4_rows; ++j){
         myfile<<dense2_weight[i][j]<<"          ";
     }
     myfile<<std::endl;
   }
   

   myfile<<"dense2_biase"<<std::endl;
   myfile<<layer5_rows<<std::endl;
   for(int i=0;i<layer5_rows; ++i){
      myfile<< dense2_biase[i]<<std::endl;
   }

   
   myfile<<"weightcnn"<<std::endl;
   myfile<<layer1_feature<<"        "<<layer1_kernel_rows*layer1_kernel_cols<<std::endl;
   for(int i=0; i<layer1_feature; ++i){
      for(int j=0; j<layer1_kernel_rows*layer1_kernel_cols; ++j){
          myfile<<weightcnn[i][j]<<"      ";
      }
      myfile<<std::endl;
   }


   myfile<<"biasecnn"<<std::endl;
   myfile<<layer1_feature<<std::endl;
   for(int i=0; i<layer1_feature; ++i){
      myfile<<biasecnn[i]<<std::endl;
   }

   
   myfile<<"history"<<std::endl;
   myfile<<Nparams<<std::endl;
   for(int i=0; i<Nparams; ++i){
       myfile<<moment[i]<<"      "<<velocity[i]<<std::endl;
   }
   
   
   
   myfile.close();
   

}
