#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<complex>
#include"H5Cpp.h"
//#include<boost/program_options.hpp>

using namespace std;

class hybrid{
  public:
	  hybrid(int fjobid, int foption, int fNtrain, int fNtest);
	  virtual ~hybrid(){}
	  void initial();
	  void read_in_training_data();
	  void read_in_hyperparameter();
          void save_parameter();

          void train_model();
	  void analyze_train();
          void test_model();

	  void initial_history_output();
	  void save_history(int datatype, int time, const string & dataname,const int & N, const std::vector<double>& A); 
          void save_history_distribution(int datatype, int time, const string & dataname,const int & N, const std::vector<long int>& A);
	  void save_analyze_result(const int &slice, const int &sample_index);
          void save_history_sample(int time);

          std::vector<long int> sort_weight, sort_gradient_weight;


          std::string history_file_name;//="history_file_"+std::to_string(jobid)+".h5";
          std::string test_out_file_name;


	  int imag_rows=28;
          int imag_cols=28;
	  int Numclass=10;
	  
	  int Nquant=9;
	  int epoch = 100;
	  int batch = 64;
	   
          int correct_number;

          double make_forward(const std::vector<std::vector<std::vector<double>>>& xin, const std::vector<std::vector<double>> &yin);
          void quantum_CNN_layer(const std::vector<std::vector<std::vector<double>>>& xin, int filter, int kernel_rows, int kernel_cols, int stride);
	  void Maxpooling2D(const std::vector<std::vector<std::vector<std::vector<double>>>> &input,  int poolsize, int stride);
	  void Flatten2D(const std::vector<std::vector<std::vector<std::vector<double>>>> &input);
	  void Dense1(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> & weights, const std::vector<double> biases);
          void Dense2(const std::vector<std::vector<double>> &input, const std::vector<std::vector<double>> & weights, const std::vector<double> biases);
	  double categoricalCrossEntropy(const std::vector<std::vector<double>> &trueProbs, const std::vector<std::vector<double>> &predictedProbs);
          int compute_true_event(const std::vector<std::vector<double>> &trueProbs, const std::vector<std::vector<double>> &predictedProbs);


	  void make_backward(const std::vector<std::vector<std::vector<double>>>& xin, const std::vector<std::vector<double>> &yin);
	  void backward_categoricalCrossEntropy(const std::vector<std::vector<double>> &trueProbs, const std::vector<std::vector<double>> &input);
	  void backward_Dense2(const std::vector<std::vector<double>> &input1, const std::vector<std::vector<double>> &input2, const std::vector<std::vector<double>> & weights);
          void backward_Dense1(const std::vector<std::vector<double>> &input1, const std::vector<std::vector<double>> &input2, const std::vector<std::vector<double>> & weights);

	  void backward_Flatten(int batch_size, int filter, int rows, int cols);
	  void backward_Maxpooling2D(const std::vector<std::vector<std::vector<std::vector<double>>>> &input,  int poolsize, int stride);
	  void backward_quantum_CNN(const std::vector<std::vector<std::vector<double>>>& xin,int filter, int kernel_rows, int kernel_cols, int stride);
	  
	  void classic_CNN_layer(const std::vector<std::vector<std::vector<double>>>& xin,int filter, int kernel_rows, int kernel_cols, int stride);
	  void backward_classic_CNN(const std::vector<std::vector<std::vector<double>>>& xin,int filter, int kernel_rows, int kernel_cols, int stride); 

	  double quantum_filter(const int NQC, std::vector<double>&x, std::vector<double>&weights, std::vector<double> &tuning);
	  
          void pack_weight();
	  void package_gradients();
          void unpackage_weights();
	  void adam_optimizer(int p);
          void save_data();
  private:
	  int jobid, choice, Ntrain, Ntest;

	  std::vector<std::vector<std::vector<double>>> train_data, test_data;
	  std::vector<std::vector<double>> y_train, y_test;

	  std::vector<double> parameters, moment, velocity, gradient;
	  
	  std::vector<std::vector<double>> weightcnn;
	  std::vector<double> biasecnn;
	  std::vector<std::vector<double>> dense1_weight;
	  std::vector<double> dense1_biase;

	  std::vector<std::vector<double>> dense2_weight;
	  std::vector<double> dense2_biase;

	  // forward layers
	  
	  int layer1_stride, layer1_rows, layer1_cols, layer1_feature, layer1_kernel_rows, layer1_kernel_cols;
	  std::vector<std::vector<std::vector<std::vector<double>>>> layer1;
	  
	  int layer2_stride, layer2_rows, layer2_cols;
	  std::vector<std::vector<std::vector<std::vector<double>>>> layer2;
	  
	  int layer3_rows;
	  std::vector<std::vector<double>> layer3;
	  
	  int layer4_rows;
	  std::vector<std::vector<double>> layer4;

          int layer5_rows;
          std::vector<std::vector<double>> layer5;	  
          std::vector<std::vector<double>> layer5_mid;	 

	  // backward layers
          std::vector<std::vector<double>> back_layer6_neurons;
	  std::vector<std::vector<double>> back_layer5_neurons;
          std::vector<std::vector<std::vector<double>>> back_layer5_weight;
	  std::vector<std::vector<double>> back_layer5_biase;

	  std::vector<std::vector<double>> back_layer4_neurons;
	  std::vector<std::vector<std::vector<double>>> back_layer4_weight;
	  std::vector<std::vector<double>> back_layer4_biase;
	  std::vector<std::vector<std::vector<std::vector<double>>>> back_layer3_neurons;
	  std::vector<std::vector<std::vector<std::vector<double>>>> back_layer2_neurons;
	  std::vector<std::vector<std::vector<double>>> back_layer1_weight;
	  std::vector<std::vector<double>> back_layer1_biase; 

	  // save history of one sample
          bool keep_sample;
	  std::vector<std::vector<double>> sample_layer1;
	  std::vector<std::vector<std::vector<double>>> sample_gradient_layer1;


	  std::string CNNmode;
	  
	  int Nparams;
	  // optimizer parameters
	  double alpha = 0.003;   // learing rate
	  double beta1=0.9;
	  double beta2=0.999;
	  double epsilon= 1e-7;
	  
	  int myrank, nproc;
	  int subbatch_size, batchbegin, batchend;
	  
};
