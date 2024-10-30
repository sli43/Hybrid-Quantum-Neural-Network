#include<iostream>
#include<fstream>
#include <math.h>
#include<complex>
#include<cmath>
#include<vector>
using namespace std;

void hybrid::package_gradients(){

    std::fill(gradient.begin(),gradient.end(),0.0); 
    int batch_size=back_layer1_weight.size();
    int filter=back_layer1_weight[0].size();
    int rows=back_layer1_weight[0][0].size();
  
    int label_index;
    int count=0;
    for(int l=0; l<filter; l++){
      for(int i=0; i<rows; i++){
         for(int k=0; k<1; k++){
             gradient[count]+=back_layer1_weight[k][l][i]/float(batch_size);
         }
         count++;
      }
    }
      

   for(int l=0; l<filter; l++){
      for(int k=0; k<1; ++k){
            gradient[count]+=back_layer1_biase[k][l]/float(batch_size);
      }
      count++;
   } 
    
    rows=back_layer4_weight[0].size();
    int cols=back_layer4_weight[0][0].size();
    for(int i=0; i<rows; i++){
      for(int j=0; j<cols; j++){
         for(int k=0; k<1; k++){
             gradient[count]+=back_layer4_weight[k][i][j]/float(batch_size);
         }
         count++;  
      }
    }
    
    for(int i=0; i<rows; i++){
       for(int k=0; k<1; k++){
          gradient[count]+=back_layer4_biase[k][i]/float(batch_size);
       }
       count++;
    }

    rows=back_layer5_weight[0].size();
    cols=back_layer5_weight[0][0].size();
    for(int i=0; i<rows; i++){
      for(int j=0; j<cols; j++){
         for(int k=0; k<1; k++){
             gradient[count]+=back_layer5_weight[k][i][j]/float(batch_size);
         }
         count++;
      }
    }

    for(int i=0; i<rows; i++){
       for(int k=0; k<1; k++){
          gradient[count]+=back_layer5_biase[k][i]/float(batch_size);
       }
       count++;
    }



}


void hybrid:: unpackage_weights(){
     int count=0;
     
     
     for(int i=0; i<layer1_feature; i++){
        for(int j=0; j<layer1_kernel_rows*layer1_kernel_cols; j++){
               weightcnn[i][j]=parameters[count];
               count++;
        }
    }

    for(int i=0; i<layer1_feature; i++){
        biasecnn[i]=parameters[count];
	count++;
    }
    
    for(int i=0; i<layer4_rows; i++){
       for(int j=0; j<layer3_rows; j++){
            dense1_weight[i][j]=parameters[count];
            count++;
       }
    } 
   
   
    for(int i=0; i<layer4_rows; i++){
        dense1_biase[i]=parameters[count];
        count++;
    }

    for(int i=0; i<layer5_rows; i++){
       for(int j=0; j<layer4_rows; j++){
            dense2_weight[i][j]=parameters[count];
            count++;
       }
    }


    for(int i=0; i<layer5_rows; i++){
        dense2_biase[i]=parameters[count];
        count++;
    }


}

void hybrid:: pack_weight(){
     int count=0;
     for(int i=0; i<layer1_feature; i++){
        for(int j=0; j<layer1_kernel_rows*layer1_kernel_cols; j++){
                parameters[count]=weightcnn[i][j];
               count++;
        }
    }

    for(int i=0; i<layer1_feature; i++){
        parameters[count]=biasecnn[i];
        count++;
    }

    for(int i=0; i<layer4_rows; i++){
       for(int j=0; j<layer3_rows; j++){
            parameters[count]=dense1_weight[i][j];
            count++;
       }
    }


    for(int i=0; i<layer4_rows; i++){
        parameters[count]=dense1_biase[i];
        count++;
    }

    for(int i=0; i<layer5_rows; i++){
       for(int j=0; j<layer4_rows; j++){
            parameters[count]=dense2_weight[i][j];
            count++;
       }
    }


    for(int i=0; i<layer5_rows; i++){
        parameters[count]=dense2_biase[i];
        count++;
    }


}

void hybrid::adam_optimizer(int iter){
     
        int t;
	t=iter+1;
	if(myrank==0){
           package_gradients();
	   for(int i=0; i<Nparams; i++){
               moment[i]   = beta1*moment[i] + (1.0 - beta1)*gradient[i];
	       velocity[i] = beta2*velocity[i] + (1.0 - beta2)*gradient[i]*gradient[i];

	       double m_hat = moment[i]/(1.0 - std::pow(beta1, t));
	       double v_hat = velocity[i]/(1.0 - std::pow(beta2, t));
	       parameters[i] = parameters[i] - alpha * m_hat/(std::sqrt(v_hat)+epsilon);
            //   parameters[i] = parameters[i] - alpha*gradient[i];
	   }
	}
	
	// passing weights
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&parameters[0],Nparams,MPI_DOUBLE,0,MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	unpackage_weights();
}
