#include<iostream>
#include<fstream>
#include<sstream>
#include<random>
#include<vector>

void hybrid:: make_backward(const std::vector<std::vector<std::vector<double>>>& xin, const std::vector<std::vector<double>> &yin){
    
     backward_categoricalCrossEntropy(yin, layer5);
     MPI_Barrier(MPI_COMM_WORLD);
     backward_Dense2(layer5,layer4,dense2_weight);
     MPI_Barrier(MPI_COMM_WORLD);
     backward_Dense1(layer4,layer3,dense1_weight);
     MPI_Barrier(MPI_COMM_WORLD);
     backward_Flatten(layer2.size(), layer2[0].size(), layer2[0][0].size(), layer2[0][0][0].size());
     MPI_Barrier(MPI_COMM_WORLD);
     backward_Maxpooling2D(layer1,2,2);
   
     MPI_Barrier(MPI_COMM_WORLD);

     if(CNNmode=="H"){
         backward_quantum_CNN(xin,layer1_feature,3,3,1);
     }else if(CNNmode=="C"){
    	 backward_classic_CNN(xin,layer1_feature,3,3,1);
     }

     MPI_Barrier(MPI_COMM_WORLD);
}


void hybrid:: backward_categoricalCrossEntropy(const std::vector<std::vector<double>> &trueProbs, const std::vector<std::vector<double>> &input){
     int batch_size=input.size();
     int rows=input[0].size();
     
     for(int k=batchbegin; k<batchend; k++){
        std::vector<double> dl_dz5(rows,0.0);
        for(int i=0; i<rows; i++){
            dl_dz5[i]=-trueProbs[k][i]/(input[k][i]+1e-15);
        }
        
        for(int i=0; i<rows; i++){
           back_layer6_neurons[k][i]=0.0;
           for(int j=0; j<rows; j++){
               double delta=0.0;
               if(i==j){delta=1.0;}         
               back_layer6_neurons[k][i]+=dl_dz5[j]*(delta*input[k][j]-input[k][i]*input[k][j]);
           }
        }
        
     }
}



void hybrid:: backward_Dense2(const std::vector<std::vector<double>> &input1, const std::vector<std::vector<double>> &input2, const std::vector<std::vector<double>> & weights){
    
     int batch_size=input1.size();
     int AFlayer_rows=input1[0].size();
     int BFlayer_rows=input2[0].size();
     
     for(int k=0; k<batch_size; ++k){
        for(int i=0; i<BFlayer_rows; ++i){
            back_layer5_neurons[k][i]=0.0;
        }
     }

     for(int k=0; k<batch_size; ++k){
        for(int i=0; i<AFlayer_rows; ++i){
             back_layer5_biase[k][i]=0.0;
	}
     }
     
     for(int k=batchbegin; k<batchend; ++k){
        for(int i=0; i<AFlayer_rows; ++i){
           for(int j=0; j<BFlayer_rows; ++j){
               back_layer5_neurons[k][j] += back_layer6_neurons[k][i]*weights[i][j];
               back_layer5_weight[k][i][j]= back_layer6_neurons[k][i]*input2[k][j];
           }
           back_layer5_biase[k][i]=back_layer6_neurons[k][i];
        }
     }

    MPI_Barrier(MPI_COMM_WORLD);
    for(int i=0; i<AFlayer_rows; ++i){
        std::vector<double> local;
	local.resize(BFlayer_rows,0.0);
	for(int j=0; j<BFlayer_rows; ++j){
            for(int k=batchbegin;k<batchend;++k){
	       local[j] +=back_layer5_weight[k][i][j];
	    }
	}
	MPI_Barrier(MPI_COMM_WORLD);
	std::vector<double> global;
	global.resize(BFlayer_rows,0.0);
	MPI_Reduce(local.data(),global.data(),BFlayer_rows,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if(myrank==0){
           for(int j=0; j<BFlayer_rows; ++j) back_layer5_weight[0][i][j]=global[j];
	}
	MPI_Barrier(MPI_COMM_WORLD);

    }
   std::vector<double> local;
   local.resize(AFlayer_rows,0.0);
   for(int j=0; j<AFlayer_rows; ++j){
     for(int k=batchbegin; k<batchend; ++k){
        local[j] +=back_layer5_biase[k][j];
     }
   }
   std::vector<double> global;
   global.resize(AFlayer_rows,0.0);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Reduce(local.data(),global.data(),AFlayer_rows,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   if(myrank==0){
       for(int j=0; j<AFlayer_rows; ++j) back_layer5_biase[0][j]=global[j];
   }
   MPI_Barrier(MPI_COMM_WORLD);

}


void hybrid:: backward_Dense1(const std::vector<std::vector<double>> &input1, const std::vector<std::vector<double>> &input2, const std::vector<std::vector<double>> & weights){
    
     int batch_size=input1.size();
     int AFlayer_rows=input1[0].size();
     int BFlayer_rows=input2[0].size();

     for(int k=0; k<batch_size; ++k){
        for(int i=0; i<BFlayer_rows; ++i){
            back_layer4_neurons[k][i]=0.0;
        }
     }

     for(int k=0; k<batch_size; ++k){
        for(int i=0; i<AFlayer_rows; ++i){
             back_layer4_biase[k][i]=0.0;
	}
     }

     for(int k=batchbegin; k<batchend; ++k){
        for(int i=0; i<AFlayer_rows; ++i){
	   double y=0.0;
	   for(int j=0; j<BFlayer_rows; ++j) y +=weights[i][j]*input2[k][j];
	   y+=dense1_biase[i];
	   double y2=4.0/(std::exp(y)+std::exp(-y))/(std::exp(y)+std::exp(-y));
	   for(int j=0; j<BFlayer_rows; ++j){
               back_layer4_neurons[k][j] += back_layer5_neurons[k][i]*y2*weights[i][j];
               back_layer4_weight[k][i][j]= back_layer5_neurons[k][i]*y2*input2[k][j];
           }
           back_layer4_biase[k][i]=back_layer5_neurons[k][i]*y2;
        }
     }

    MPI_Barrier(MPI_COMM_WORLD);
    for(int i=0; i<AFlayer_rows; ++i){
        std::vector<double> local;
        local.resize(BFlayer_rows,0.0);
        for(int j=0; j<BFlayer_rows; ++j){
            for(int k=batchbegin;k<batchend;++k){
               local[j] +=back_layer4_weight[k][i][j];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<double> global;
        global.resize(BFlayer_rows,0.0);
        MPI_Reduce(local.data(),global.data(),BFlayer_rows,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if(myrank==0){
           for(int j=0; j<BFlayer_rows; ++j) back_layer4_weight[0][i][j]=global[j];
        }
        MPI_Barrier(MPI_COMM_WORLD);

    }
   std::vector<double> local;
   local.resize(AFlayer_rows,0.0);
   for(int j=0; j<AFlayer_rows; ++j){
     for(int k=batchbegin; k<batchend; ++k){
        local[j] +=back_layer4_biase[k][j];
     }
   }
   std::vector<double> global;
   global.resize(AFlayer_rows,0.0);
   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Reduce(local.data(),global.data(),AFlayer_rows,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
   if(myrank==0){
       for(int j=0; j<AFlayer_rows; ++j) back_layer4_biase[0][j]=global[j];
   }
   MPI_Barrier(MPI_COMM_WORLD);



}



void hybrid:: backward_Flatten(int batch_size, int filter, int rows, int cols){
     
     for(int k=batchbegin; k<batchend; ++k){
            int count=0;
            for(int l=0; l<filter; ++l){
               for(int i=0; i<rows; ++i){
                  for(int j=0; j<cols; ++j){
	              if(CNNmode=="H"){
                           back_layer3_neurons[k][l][i][j]=back_layer4_neurons[k][count];
		      } else if(CNNmode=="C"){
                           back_layer3_neurons[k][l][i][j]=back_layer4_neurons[k][count];
		      }
		      
		      count=count+1;
                  }
               }
            }
     }
     
}



void hybrid:: backward_Maxpooling2D(const std::vector<std::vector<std::vector<std::vector<double>>>> &input,  int poolsize, int stride){
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
                  for(int x=0; x<poolsize; ++x){
                     for(int y=0; y<poolsize; ++y){
                        int inputRow = i*stride +x;
                        int inputCol = j*stride +y;
	                     back_layer2_neurons[k][l][inputRow][inputCol]=back_layer3_neurons[k][l][i][j]/float(poolsize*poolsize);
                     }
                  }
               }
            }
         
         }
      
      }      

}


void hybrid:: backward_classic_CNN(const std::vector<std::vector<std::vector<double>>>& xin,int filter, int kernel_rows, int kernel_cols, int stride){
   int batch_size=xin.size();
   int rows=xin[0].size();
   int cols=xin[0][0].size();

   std::vector<double> x(Nquant);
   std::vector<double> w(Nquant);

   for(int k=0; k<batch_size; ++k){
     for(int l=0; l<filter; ++l){
       for(int iw=0; iw<Nquant; ++iw){
          back_layer1_weight[k][l][iw]=0.0;
       }
       back_layer1_biase[k][l]=0.0;
     }
   }

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
		for(int iw=0; iw<Nquant; ++iw) w[iw]=weightcnn[l][iw];
		double p=tanh_derivative(Nquant,x,w,biasecnn[l]);
		for(int iw=0; iw<Nquant; ++iw){
		    back_layer1_weight[k][l][iw]+=back_layer2_neurons[k][l][kx][ky]*p*x[iw];
		    if(k==0&&keep_sample){
                             sample_gradient_layer1[l][kx*Ly+ky][iw]=p*x[iw];
                    }
                    int fi=int(abs(p*x[iw]/0.05));
		    if(fi>20) fi=20;
		    sort_gradient_weight[fi] +=1;
		}
		back_layer1_biase[k][l]+=back_layer2_neurons[k][l][kx][ky]*p;
            }
        }
     }
   }

    MPI_Barrier(MPI_COMM_WORLD);
    for(int l=0; l<filter; ++l){
        std::vector<double> local;
        local.resize(Nquant,0.0);
        for(int j=0; j<Nquant; ++j){
            for(int k=batchbegin;k<batchend;++k){
               local[j] +=back_layer1_weight[k][l][j];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<double> global;
        global.resize(Nquant,0.0);
        MPI_Reduce(local.data(),global.data(),Nquant,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if(myrank==0){
           for(int j=0; j<Nquant; ++j) back_layer1_weight[0][l][j]=global[j];
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

   MPI_Barrier(MPI_COMM_WORLD);
   std::vector<double> local;
   local.resize(filter,0.0);
   for(int l=0; l<filter; ++l){
       for(int k=0; k<batch_size; ++k){
          local[l] +=back_layer1_biase[k][l];
       }
   }
   std::vector<double> global;
   global.resize(filter,0.0);
   MPI_Reduce(local.data(), global.data(), filter, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
   if(myrank==0){
        for(int l=0; l<filter; ++l) back_layer1_biase[0][l]=global[l];
   }
   MPI_Barrier(MPI_COMM_WORLD);


}

void hybrid:: backward_quantum_CNN(const std::vector<std::vector<std::vector<double>>>& xin,int filter, int kernel_rows, int kernel_cols, int stride){
     
   int batch_size=xin.size();
   int rows=xin[0].size();
   int cols=xin[0][0].size();

   std::vector<double> x(Nquant);
   std::vector<double> w(Nquant);
   
   for(int k=0; k<batch_size; ++k){
     for(int l=0; l<filter; ++l){
       for(int iw=0; iw<Nquant; ++iw){
          back_layer1_weight[k][l][iw]=0.0;
       }
     }
   }
   
   
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
                    for(int iw=0; iw<Nquant; ++iw) w[iw]=weightcnn[l][iw];
                    for(int iw=0; iw<Nquant; ++iw){
                         std::vector<double> tuning(Nquant,0.0);
                         std::fill(tuning.begin(),tuning.end(),0.0);
                         tuning[iw]=0.5;
                         double fy1=quantum_filter(Nquant,x,w, tuning);
                         tuning[iw]=-0.5;
                         double fy2=quantum_filter(Nquant,x,w, tuning);
                         back_layer1_weight[k][l][iw]+=0.5*(fy1-fy2)*back_layer2_neurons[k][l][kx][ky];
                         if(k==0&&keep_sample){
                             sample_gradient_layer1[l][kx*Ly+ky][iw]=0.5*(fy1-fy2); 
			 }
			 int fi=0;
			 fi=int( abs((fy1-fy2)*0.5/0.05) );
		         sort_gradient_weight[fi] +=1;
 		    }
            }
        }
     }    
   }

    MPI_Barrier(MPI_COMM_WORLD);
    for(int l=0; l<filter; ++l){
        std::vector<double> local;
        local.resize(Nquant,0.0);
        for(int j=0; j<Nquant; ++j){
            for(int k=batchbegin;k<batchend;++k){
               local[j] +=back_layer1_weight[k][l][j];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<double> global;
        global.resize(Nquant,0.0);
        MPI_Reduce(local.data(),global.data(),Nquant,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if(myrank==0){
           for(int j=0; j<Nquant; ++j) back_layer1_weight[0][l][j]=global[j];
        }
        MPI_Barrier(MPI_COMM_WORLD);

    }
}

