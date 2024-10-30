#include "mpi.h"
#include<iostream>
#include<fstream>
#include<random>
#include<vector>

void hybrid:: train_model(){
        if(myrank==0) initial_history_output();
       // assert(nproc==Nquant);
        MPI_Barrier(MPI_COMM_WORLD);
//        read_in_hyperparameter();
	std::fill(moment.begin(),moment.end(),0.0);
        std::fill(velocity.begin(),velocity.end(),0.0);
        std::fill(gradient.begin(),gradient.end(),0.0);

        std::string loss_file_name;
        if(CNNmode=="C"){
             loss_file_name="Closs_"+std::to_string(jobid)+".dat";
        }else if(CNNmode=="H"){
             loss_file_name="Hloss_"+std::to_string(jobid)+".dat";
        }
        std::ofstream loss_file(loss_file_name);

        std::vector<std::vector<std::vector<double>>> xbatch;
        xbatch.resize(batch,std::vector<std::vector<double>>(imag_rows,std::vector<double>(imag_cols,0.0)));
        std::vector<std::vector<double>> ybatch;
        ybatch.resize(batch,std::vector<double>(Numclass,0.0));
        double loss;
        double accuracy;
        int iter=0;
        for(int i=0; i<epoch; i++){
           if(myrank==0) {std::cout<<"epoch "<<i<<"/"<<epoch<<std::endl;}
           double total_loss=0.0;
           int total_correct_number=0;
           std::fill(moment.begin(),moment.end(),0.0);
           std::fill(velocity.begin(),velocity.end(),0.0);
      
           std::fill(sort_weight.begin(), sort_weight.end(),0);
	   std::fill(sort_gradient_weight.begin(),sort_gradient_weight.end(),0);
           iter=0;
	   for(int p=0; p<Ntrain/batch; p++){
               if(myrank==0) {std::cout<<"ith batch = "<<p<<"/"<<Ntrain/batch<<std::endl;}
               for(int k=0; k<batch; k++){
                  for(int kx=0; kx<imag_rows; kx++){
                    for(int ky=0; ky<imag_cols; ky++){
                        xbatch[k][kx][ky]=train_data[p*batch+k][kx][ky];
                    }
                  }
                  for(int y=0; y<Numclass; y++) ybatch[k][y]=y_train[p*batch+k][y];
               }
               if(p==0) {keep_sample=true;}
	       else{ keep_sample=false; }

               loss = make_forward(xbatch, ybatch);
               if(myrank==0){
                  total_loss+=loss;
                  total_correct_number+=correct_number;
                  std::cout<<"loss = "<<loss/float(batch)<<"    accuracy =  "<<correct_number/float(batch)<<std::endl;
               }
               MPI_Barrier(MPI_COMM_WORLD);

               make_backward(xbatch, ybatch);
               MPI_Barrier(MPI_COMM_WORLD);
	       adam_optimizer(iter);
               MPI_Barrier(MPI_COMM_WORLD);
               iter++;
	       if(myrank==0&&p<1){
                   save_history(1, i*Ntrain/batch+iter, "gradient", Nparams, gradient);
                   save_history(2, i*Ntrain/batch+iter, "weight", Nparams, parameters);
                   save_history(3, i*Ntrain/batch+iter, "moment", Nparams, moment);
		   save_history(4, i*Ntrain/batch+iter, "velocity",Nparams, velocity);
	       }
           }

           MPI_Barrier(MPI_COMM_WORLD);
           for(int k=0; k<21; ++k){
               long int sum;
	       MPI_Reduce(&sort_weight[k],&sum,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
	       if(myrank==0) sort_weight[k]=sum;
	       MPI_Barrier(MPI_COMM_WORLD);
	   }

	   MPI_Barrier(MPI_COMM_WORLD);
           for(int k=0; k<21; ++k){
               long int sum;
               MPI_Reduce(&sort_gradient_weight[k],&sum,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
               if(myrank==0) sort_gradient_weight[k]=sum;
               MPI_Barrier(MPI_COMM_WORLD);
           }


	   if(myrank==0){
                   accuracy=total_correct_number/float(Ntrain);
                   std::cout<<"total loss = "<<total_loss/float(Ntrain)<<"      total accuracy = "<<total_correct_number/float(Ntrain)<<std::endl;
                   loss_file<<i<<"      "<<total_loss/float(Ntrain)<<"          "<<total_correct_number/float(Ntrain)<<std::endl;
           }


        if(myrank==0) {std::cout<<"==========================================================\n";}
        if(myrank==0) {
             //   save_history(1, i, "gradient", Nparams, gradient);
             //   save_history(2, i, "weight", Nparams, parameters);
                save_history_distribution(1,i,"gradient",21,sort_gradient_weight);
		save_history_distribution(2,i,"weight",21,sort_weight);
                save_history_sample(i);
      		save_data();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&accuracy, 1, MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if(accuracy>0.9999) break;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(myrank==0) save_parameter();
	if(myrank==0) loss_file.close();

}
