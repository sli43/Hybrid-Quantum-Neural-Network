#include<iostream>
#include<fstream>
#include<sstream>
#include<random>
#include<vector>
//#include<boost/program_options.hpp>

double hybrid::make_forward(const std::vector<std::vector<std::vector<double>>>& xin, const std::vector<std::vector<double>> &yin){
     
     int batch_size=xin.size();  
     subbatch_size=batch_size/nproc;
     batchbegin=myrank*subbatch_size;
     batchend=(myrank+1)*subbatch_size;
     if(myrank==nproc-1){ batchend=batch_size;}
     subbatch_size=batchend-batchbegin;
     
     if(CNNmode=="C"){
       classic_CNN_layer(xin,layer1_feature,3,3,1);}
     else if(CNNmode=="H"){   
        quantum_CNN_layer(xin,layer1_feature,3,3,1);
     }
     MPI_Barrier(MPI_COMM_WORLD); 
     Maxpooling2D(layer1,2,2);
     MPI_Barrier(MPI_COMM_WORLD); 
     Flatten2D(layer2);
     MPI_Barrier(MPI_COMM_WORLD); 
     Dense1(layer3,dense1_weight, dense1_biase);
     MPI_Barrier(MPI_COMM_WORLD);
     Dense2(layer4,dense2_weight, dense2_biase);
     MPI_Barrier(MPI_COMM_WORLD); 
     
     double loss, sumloss;
     sumloss=0.0;
     loss=categoricalCrossEntropy(yin, layer5);
     int sum_correct_num;
     correct_number=compute_true_event(yin, layer5);
     sum_correct_num=0;

     MPI_Barrier(MPI_COMM_WORLD);   
     MPI_Reduce(&loss, &sumloss, 1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
     MPI_Barrier(MPI_COMM_WORLD);   
     MPI_Reduce(&correct_number, &sum_correct_num, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
     correct_number=sum_correct_num;
     MPI_Barrier(MPI_COMM_WORLD);
     return sumloss;

}

void hybrid::classic_CNN_layer(const std::vector<std::vector<std::vector<double>>>& xin,int filter, int kernel_rows, int kernel_cols, int stride){

   int batch_size=xin.size();
   int rows=xin[0].size();
   int cols=xin[0][0].size();
   std::vector<double> tuning(Nquant,0.0);

   std::vector<double> x(Nquant);
   std::vector<double> w(Nquant);

   for(int k=batchbegin; k<batchend; ++k){
     int Lx=(rows-kernel_rows)/stride+1;
     int Ly=(cols-kernel_cols)/stride+1;
     for(int kx=0; kx<Lx; ++kx){
        for(int ky=0; ky<Ly; ++ky){
            int ic=0;
            for(int sx=0; sx<kernel_rows; ++sx){
              for(int sy=0; sy<kernel_cols; ++sy){
                  x[ic]=xin[k][kx*stride+sx][ky*stride+sy];
                  ic=ic+1;
              }
            }
            for(int l=0; l<filter; ++l){
                for(int iw=0; iw<Nquant; ++iw) {w[iw]=weightcnn[l][iw];}
                layer1[k][l][kx][ky]=tanh(Nquant,x,w,biasecnn[l]);
		if(k==0&&keep_sample){
                    sample_layer1[l][kx*Ly+ky]=layer1[k][l][kx][ky];
                }
                int fi=int( layer1[k][l][kx][ky]/0.05 );
		if(fi>20) fi=20;
		sort_weight[fi] +=1;
	    }
        }
     }
   }


}

void hybrid::quantum_CNN_layer(const std::vector<std::vector<std::vector<double>>>& xin,int filter, int kernel_rows, int kernel_cols, int stride){

   int batch_size=xin.size();
   int rows=xin[0].size();
   int cols=xin[0][0].size();
   std::vector<double> tuning(Nquant,0.0);

   std::vector<double> x(Nquant);
   std::vector<double> w(Nquant);
   
   for(int k=batchbegin; k<batchend; ++k){
     int Lx=(rows-kernel_rows)/stride+1;
     int Ly=(cols-kernel_cols)/stride+1;
     for(int kx=0; kx<Lx; ++kx){
        for(int ky=0; ky<Ly; ++ky){
            int ic=0;
            for(int sx=0; sx<kernel_rows; ++sx){
              for(int sy=0; sy<kernel_cols; ++sy){
                  x[ic]=xin[k][kx*stride+sx][ky*stride+sy];
                  ic=ic+1;
              }
            }
            for(int l=0; l<filter; ++l){
                for(int iw=0; iw<Nquant; ++iw) {w[iw]=weightcnn[l][iw];}
                layer1[k][l][kx][ky]=quantum_filter(Nquant,x,w,tuning);
                if(k==0&&keep_sample){
                    sample_layer1[l][kx*Ly+ky]=layer1[k][l][kx][ky]; 
		}
		int fi=int( abs(layer1[k][l][kx][ky]/0.05) );
	        if(fi>20) {fi=20;}
		sort_weight[fi] +=1;
	    }          
        }
     }    
   }
}

void hybrid:: Maxpooling2D(const std::vector<std::vector<std::vector<std::vector<double>>>> &input,  int poolsize, int stride){

      int batch_size=input.size();
      int filter=input[0].size();
      int rows = input[0][0].size();
      int cols = input[0][0][0].size();
      int outputRows = (rows + stride-1)/stride;
      int outputCols = (cols + stride -1)/stride;
      
      
      for(int k=batchbegin; k<batchend; ++k){
         for(int l=0; l<filter; ++l){
            for(int i=0; i<outputRows; ++i){
               for(int j=0; j<outputCols; ++j){
                  double maxval = 0.0;//input[k][l][i*stride][j*stride];
                  for(int x=0; x<poolsize; ++x){
                     for(int y=0; y<poolsize; ++y){
                        int inputRow = i*stride +x;
                        int inputCol = j*stride +y;
		          maxval+=input[k][l][inputRow][inputCol];
                     }
                  }
                  layer2[k][l][i][j]=maxval/float(poolsize*poolsize);
 	       }
            }
         
         }
      
      }
      
 }
  
  
 void hybrid:: Flatten2D(const std::vector<std::vector<std::vector<std::vector<double>>>> &input){
            
         
      int batch_size=input.size();
      int filter = input[0].size();
      int rows = input[0][0].size();
      int cols = input[0][0][0].size();
      
      int outputRows = filter*rows*cols;
      
      for(int k=batchbegin; k<batchend; ++k){
         int count=0;
         for(int l=0; l<filter;++l){
            for(int i=0; i<rows; ++i){
               for(int j=0; j<cols; ++j){
		  if(CNNmode=="H"){
                  layer3[k][count]=input[k][l][i][j];
		  }else if(CNNmode=="C"){
                    layer3[k][count]=input[k][l][i][j];
		  }
                  count+=1;
               }
            }
         }     
      }
 
 }     
      

void hybrid:: Dense1(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> & weights, const std::vector<double> biases){
    int batch_size= input.size();
    int inputsize = input[0].size();
    int outputsize = biases.size();

    for(int k=batchbegin; k<batchend; ++k){
       for(int i=0; i< outputsize; ++i){
         double y=0.0;
      	 for(int j=0; j<inputsize; ++j){
             y+=weights[i][j]*input[k][j];
         }
         y+=biases[i];
	 layer4[k][i]=(std::exp(2*y)-1)/(std::exp(2*y)+1);
       }

    }

}

void hybrid:: Dense2(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> & weights, const std::vector<double> biases){
    int batch_size= input.size();
    int inputsize = input[0].size();
    int outputsize = biases.size();
    
    for(int k=batchbegin; k<batchend; ++k){
       std::vector<double> z(outputsize,0.0);
       for(int i=0; i< outputsize; ++i){
         for(int j=0; j<inputsize; ++j){
             z[i]+=weights[i][j]*input[k][j];
         }
         z[i]+=biases[i];
	 layer5_mid[k][i]=z[i];
       }
       
       std::vector<double> output(outputsize,0.0);
       output=softmax(z);
       for(int i=0; i<outputsize; ++i){ layer5[k][i]=output[i]; 
       }
    }

}



double hybrid:: categoricalCrossEntropy(const std::vector<std::vector<double>> &trueProbs, const std::vector<std::vector<double>> &predictedProbs){

    int batch_size=trueProbs.size();
    int class_size=trueProbs[0].size();
    double loss=0.0;
    for(int k=batchbegin; k<batchend; ++k){
       for(int i=0; i<class_size; ++i){
          loss-=trueProbs[k][i]*log(predictedProbs[k][i]+1e-15);
       }
    }
    return loss;
}

int hybrid:: compute_true_event(const std::vector<std::vector<double>> &trueProbs, const std::vector<std::vector<double>> &predictedProbs){

    int count=0;
    int class_size=trueProbs[0].size();

    for(int k=batchbegin; k<batchend; ++k){
       int ip=0;
       double maxval=predictedProbs[k][0];
       for(int i=0; i<10; ++i){
          if(maxval<predictedProbs[k][i]){
             maxval=predictedProbs[k][i];
	     ip=i;
	  }
       }

       int jp=0;
       maxval=trueProbs[k][0];
       for(int i=0; i<10; ++i){
          if(maxval<trueProbs[k][i]){
             maxval=trueProbs[k][i];
	     jp=i;
	  }
       }

      if(ip==jp) {count=count+1;}

    }

    return count;
}
