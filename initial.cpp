#include<iostream>
#include<fstream>
#include<random>
#include<vector>

void hybrid:: initial(){
        
       MPI_Comm_size(MPI_COMM_WORLD, &nproc);
       MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
       if(myrank==0){ 
             std::cout<<"==============================================="<<std::endl;
             std::cout<<"input image: (rows, cols) = "<<imag_rows<<"  "<<imag_cols<<std::endl;
       }
       CNNmode="H";

       sort_weight.resize(21,0);
       sort_gradient_weight.resize(21,0);
       history_file_name="history_file_"+std::to_string(jobid)+".h5";
       test_out_file_name="test_file_"+std::to_string(jobid)+".dat";
       // initial layers
       layer1_stride=1;
       layer1_kernel_rows=3;
       layer1_kernel_cols=3;
       layer1_feature=16;
       layer1_rows=(imag_rows-layer1_kernel_rows)/layer1_stride+1;
       layer1_cols=(imag_cols-layer1_kernel_cols)/layer1_stride+1;
        layer1.resize(batch,std::vector<std::vector<std::vector<double>>>
        (layer1_feature,std::vector<std::vector<double>>
        (layer1_rows,std::vector<double>(layer1_cols,0.0))));
        
        if(myrank==0){std::cout<<"Layer 1: "<<layer1_feature<<"   "<< layer1_rows<<"   "<<layer1_cols<<std::endl;}
        
        layer2_stride=2;
        layer2_rows = (layer1_rows+layer2_stride-1)/layer2_stride;
        layer2_cols = (layer1_cols+layer2_stride-1)/layer2_stride;
        
        layer2.resize(batch,std::vector<std::vector<std::vector<double>>>
        (layer1_feature,std::vector<std::vector<double>>
        (layer2_rows,std::vector<double>(layer2_cols,0.0))));
        
        if(myrank==0) {std::cout<<"Layer 2: "<<layer1_feature<<"   "<< layer2_rows<<"   "<<layer2_cols<<std::endl;}
        
       layer3_rows=layer1_feature*layer2_rows*layer2_cols;
       layer3.resize(batch,std::vector<double>(layer3_rows,0.0));
        if(myrank==0) {std::cout<<"Layer 3: "<< layer3_rows<<std::endl;}
        
	layer4_rows=32;
	layer4.resize(batch,std::vector<double>(layer4_rows,0.0));
	if(myrank==0) {std::cout<<"Layer 4: "<< layer4_rows<<std::endl;}
        
        layer5_rows=Numclass;
        layer5.resize(batch,std::vector<double>(layer5_rows,0.0));
        layer5_mid.resize(batch,std::vector<double>(layer5_rows,0.0));
   	if(myrank==0) {std::cout<<"Layer 5: "<< layer5_rows<<std::endl;}
        
        // keep sample layers 
        sample_layer1.resize(layer1_feature,std::vector<double>(layer1_rows*layer1_cols,0.0));
	sample_gradient_layer1.resize(layer1_feature,std::vector<std::vector<double>>(layer1_rows*layer1_cols,std::vector<double>(Nquant,0.0)));


        // backward layers
        back_layer6_neurons.resize(batch,std::vector<double>(layer5_rows,0.0));
     
     	back_layer5_neurons.resize(batch,std::vector<double>(layer4_rows,0.0));
	back_layer5_weight.resize(batch,std::vector<std::vector<double>>(layer5_rows,std::vector<double>(layer4_rows,0.0)));
	back_layer5_biase.resize(batch,std::vector<double>(layer5_rows,0.0));

   	back_layer4_neurons.resize(batch,std::vector<double>(layer3_rows,0.0));
        back_layer4_weight.resize(batch,std::vector<std::vector<double>>(layer4_rows,std::vector<double>(layer3_rows,0.0)));
        back_layer4_biase.resize(batch,std::vector<double>(layer4_rows,0.0));
     
     	back_layer3_neurons.resize(batch,std::vector<std::vector<std::vector<double>>>
        (layer1_feature,std::vector<std::vector<double>>
        (layer2_rows,std::vector<double>(layer2_cols,0.0))));
        
        back_layer2_neurons.resize(batch,std::vector<std::vector<std::vector<double>>>
        (layer1_feature,std::vector<std::vector<double>>(layer1_rows,std::vector<double>
        (layer1_cols,0.0))));
        
        back_layer1_weight.resize(batch,std::vector<std::vector<double>>(layer1_feature,std::vector<double>(Nquant,0.0)));
        back_layer1_biase.resize(batch,std::vector<double>(layer1_feature,0.0)); 
               
       // initial weights
        int total_num_parameters=0;
        std::random_device rd;
        std::mt19937 gen(rd());
        
        weightcnn.resize(layer1_feature,std::vector<double>(layer1_feature*layer1_kernel_rows*layer1_kernel_cols,0.0));
        if(CNNmode=="C"){       
	      std::uniform_real_distribution<double> dist_double(-1, 1);
              for(int i=0; i<layer1_feature; ++i){
                 for(int j=0; j<layer1_kernel_rows*layer1_kernel_cols;++j){
                 weightcnn[i][j]=dist_double(gen);
                 }
              }
	}else{
             std::uniform_real_distribution<double> dist_double(0, 2);
             for(int i=0; i<layer1_feature; ++i){
               for(int j=0; j<layer1_kernel_rows*layer1_kernel_cols;++j){
                 weightcnn[i][j]=dist_double(gen);
               }
             }
	}


	biasecnn.resize(layer1_feature,0.0);
        if(CNNmode=="C"){
            std::uniform_real_distribution<double> uniform_double(-1, 1);
            for(int i=0; i<layer1_feature; ++i){
                 biasecnn[i]=uniform_double(gen);
	    }
	}

        total_num_parameters+=layer1_feature*layer1_kernel_rows*layer1_kernel_cols+layer1_feature;
        
        dense1_weight.resize(layer4_rows, std::vector<double>(layer3_rows,0.0));
        dense1_biase.resize(layer4_rows,0.0);
        std::uniform_real_distribution<double> uniform_double(-1, 1);
        for (int i=0; i<layer4_rows; ++i){
             for(int j=0; j<layer3_rows; ++j) { dense1_weight[i][j]=uniform_double(gen); }
             dense1_biase[i]=uniform_double(gen);
        }  
        total_num_parameters+=layer4_rows*layer3_rows + layer4_rows;
        

        dense2_weight.resize(layer5_rows,std::vector<double>(layer4_rows,0.0));
	dense2_biase.resize(layer5_rows,0.0);
	for(int i=0; i<layer5_rows;++i){
           for(int j=0; j<layer4_rows;++j){
               dense2_weight[i][j]=uniform_double(gen);
	   }
	   dense2_biase[i]=uniform_double(gen);
	}
	total_num_parameters+=layer5_rows*layer4_rows + layer5_rows;

        // passing parameters
        MPI_Barrier(MPI_COMM_WORLD);
        int k=0;
        std::vector<double> flattenedvector(layer1_feature*layer1_kernel_rows*layer1_kernel_cols,0.0);
        for(int i=0; i<layer1_feature; ++i){
           for(int j=0; j<layer1_kernel_rows*layer1_kernel_cols; ++j){
               flattenedvector[k]=weightcnn[i][j];
               k++;
           }
        }
        MPI_Bcast(&flattenedvector[0],layer1_feature*layer1_kernel_rows*layer1_kernel_cols,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        
        k=0;
        for(int i=0; i<layer1_feature; ++i){
          for(int j=0; j<layer1_kernel_rows*layer1_kernel_cols; ++j){
             weightcnn[i][j]=flattenedvector[k];
             k++;
          }
        }
    
        MPI_Bcast(&biasecnn[0],layer1_feature,MPI_DOUBLE,0,MPI_COMM_WORLD);	
        
        k=0;
        flattenedvector.resize(layer4_rows*layer3_rows,0.0);
        for(int i=0; i<layer4_rows; ++i){
           for(int j=0; j<layer3_rows; ++j){
               flattenedvector[k]=dense1_weight[i][j];
               k++;
           }
        }
        
        MPI_Bcast(&flattenedvector[0], layer4_rows*layer3_rows, MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        k=0;
        for(int i=0; i<layer4_rows; ++i){
           for(int j=0; j<layer3_rows; ++j){
               dense1_weight[i][j]=flattenedvector[k];
               k++;
           }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        MPI_Bcast(&dense1_biase[0], layer4_rows, MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
       
        k=0;
        flattenedvector.resize(layer5_rows*layer4_rows,0.0);
        for(int i=0; i<layer5_rows; ++i){
           for(int j=0; j<layer4_rows; ++j){
               flattenedvector[k]=dense2_weight[i][j];
               k++;
           }
        }

        MPI_Bcast(&flattenedvector[0], layer5_rows*layer4_rows, MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        k=0;
        for(int i=0; i<layer5_rows; ++i){
           for(int j=0; j<layer4_rows; ++j){
               dense2_weight[i][j]=flattenedvector[k];
               k++;
           }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Bcast(&dense2_biase[0], layer5_rows, MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);



       // initial optimizer
        Nparams=total_num_parameters;
	parameters.resize(Nparams,0.0);
	moment.resize(Nparams,0.0);
	velocity.resize(Nparams,0.0);
        gradient.resize(Nparams,0.0);
        std::fill(moment.begin(),moment.end(),0.0);
        std::fill(velocity.begin(),velocity.end(),0.0);
	std::fill(gradient.begin(),gradient.end(),0.0);	
        int count=0;
        for(int i=0; i<layer1_feature; ++i){
           for(int j=0; j<layer1_kernel_rows*layer1_kernel_cols; ++j){
               parameters[count]=weightcnn[i][j];
               count++;
           }
        }
        
        for(int i=0; i<layer1_feature; ++i){
               parameters[count]=biasecnn[i];
	       count++;
	}

        for(int i=0; i<layer4_rows; ++i){
          for(int j=0; j<layer3_rows; ++j){
              parameters[count]=dense1_weight[i][j];
              count++;
          }
        }
        
        for(int i=0; i<layer4_rows; ++i){
            parameters[count]=dense1_biase[i];
            count++;
        }


        for(int i=0; i<layer5_rows; ++i){
          for(int j=0; j<layer4_rows; ++j){
              parameters[count]=dense2_weight[i][j];
              count++;
          }
        }

        for(int i=0; i<layer5_rows; ++i){
            parameters[count]=dense2_biase[i];
            count++;
        }	
        if(myrank==0) {std::cout<<"================================================================================\n";}
        

}
