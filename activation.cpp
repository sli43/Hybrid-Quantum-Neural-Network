#include<iostream>
#include<fstream>
#include<sstream>
#include<random>
#include<vector>

std::vector<double> softmax(const std::vector<double> &z){
   double max_z = *std::max_element(z.begin(),z.end());
   std::vector<double> e_z(z.size());
   double sum_e_z=0.0;
   
   for(int i=0; i<z.size(); i++){
      e_z[i]=std::exp(z[i]-max_z);
      sum_e_z+=e_z[i];
   }
   
   std::vector<double> p(z.size());
   for(int i=0; i<z.size(); i++){
      p[i]=e_z[i]/sum_e_z;
   }
   
   return p;
}


double sigmoid(const int N, const std::vector<double> &z, const std::vector<double> & wp, const double biase){
    double p=0.0;
    for(int i=0; i<N; ++i){
        p+=z[i]*wp[i];
    }
    p+=biase;
    p=1.0/(1.0+std::exp(-p));
    return p;
}	

double sigmoid_derivative(const int N, const std::vector<double> &z, const std::vector<double> & wp, const double biase){
   double x=0.0;
   for(int i=0; i<N; ++i){
       x+=z[i]*wp[i];
   }
   x+=biase;
   double p;
   p=std::exp(-x)/(1.0+std::exp(-x))/(1.0+std::exp(-x));
   return p;
}

double tanh(const int N, const std::vector<double> &z, const std::vector<double> & wp, const double biase){
    double x=0.0;
    for(int i=0; i<N; ++i){
        x+=z[i]*wp[i];
    }
    x+=biase;
    double p;
    p=(std::exp(2*x)-1)/(std::exp(2*x)+1);
    return p;
}

double tanh_derivative(const int N, const std::vector<double> &z, const std::vector<double> & wp, const double biase){
   double x=0.0;
   for(int i=0; i<N; ++i){
       x+=z[i]*wp[i];
   }
   x+=biase;
   double p;
   p=4.0/(std::exp(x)+std::exp(-x))/(std::exp(x)+std::exp(-x));
   return p;
}
