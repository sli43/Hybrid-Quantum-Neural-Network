#include<iostream>
#include<fstream>
#include<sstream>
#include<random>
#include<vector>
#include<boost/program_options.hpp>

#include<cppsim/state.hpp>
#include<cppsim/circuit.hpp>
#include<cppsim/observable.hpp>
#include <vqcsim/parametric_circuit.hpp>
#include <cppsim/utility.hpp>
#include <cppsim/circuit_optimizer.hpp>

# define M_PI           3.14159265358979323846  

using namespace std;
double hybrid::quantum_filter(const int NQC, std::vector<double>&x, std::vector<double>&weights, std::vector<double> &tuning){
       
       QuantumState state(NQC);
       state.set_zero_state();
       
       // create n-qubit parametric circuit
       ParametricQuantumCircuit* circuit = new ParametricQuantumCircuit(NQC); 

       // encode data
       for(int i=0; i<NQC; i++){
           circuit->add_parametric_RY_gate(i,x[i]*M_PI);
       }
       
       for(int i=0; i<NQC; i=i+2){
          if(i+1<NQC){ circuit->add_CNOT_gate(i,i+1);}
       }
       
       for(int i=1; i<NQC; i=i+2){
          if(i+1<NQC){ circuit->add_CNOT_gate(i,i+1);}
          else { circuit->add_CNOT_gate(i,0);}
       }
    
       for(int i=0; i<NQC; i++){
          circuit->add_parametric_RY_gate(i,weights[i]+tuning[i]*M_PI);
       }
 
       for(int i=0; i<NQC; i=i+2){
          if(i+1<NQC){ circuit->add_CNOT_gate(i,i+1);}
       }

       for(int i=1; i<NQC; i=i+2){
          if(i+1<NQC){ circuit->add_CNOT_gate(i,i+1);}
          else { circuit->add_CNOT_gate(i,0);}
       }


       QuantumCircuitOptimizer optimizer;
       optimizer.optimize_light(circuit);

       circuit->update_quantum_state(&state);
       
      	Observable observable(NQC);
       	std::string measurement;
       	for(int i=0; i<NQC; i++){
       	    measurement += "Z";
       	    measurement += " ";
       	    measurement +=std::to_string(i);
       	    if(i<NQC-1){ measurement +=" ";}
       	}
        observable.add_operator(1.0, measurement);
        
        std::complex<double> mean;
        mean=observable.get_expectation_value(&state);
        

        delete circuit;
       return mean.real();
}
