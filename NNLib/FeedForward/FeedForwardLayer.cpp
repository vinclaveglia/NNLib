/*
 * FeedForwordLayer.cpp
 *
 *  Created on: 29 giu 2016
 *      Author: vincenzo
 */

#include "FeedForwardLayer.h"
#include "cmath"
#include "../act_function.h"
#include "../mlmath.h"
#include <new>
#include <exception>
#include <iostream>

using namespace std;


FeedForwardLayer::FeedForwardLayer(int num_of_neurons){
    this->id = 0;
    this->num_of_neurons = num_of_neurons;
    this->isInputLayer = true;
    this->type = FEEDFORWARD;
    //è un input layer, non ha matrice dei pesi
    //l'output del layer corrisponde al suo input
}



FeedForwardLayer::FeedForwardLayer(int num_of_neurons, int previous_layer_size, int layer_id, std::string act_function){
    this->id = layer_id;
    this->num_of_neurons = num_of_neurons;
    this->num_of_inputs = previous_layer_size;
    this->l_act_function = act_function;
    this->type = FEEDFORWARD;

    cout << "Activation function :: " << l_act_function << endl;

    //alloca matrice dei pesi
    connections = new float*[num_of_neurons];
    for(int i=0; i<num_of_neurons;  i++){
        //ogni puntatore punta a una array
        connections[i] = new float[previous_layer_size];
    }

    this->bias = new float[num_of_neurons];

    this->delta = new float[num_of_neurons];
    this->a = new float[num_of_neurons];
    this->out = new float[num_of_neurons];
    this->error = new float[num_of_neurons];//in realtà questo è necessario solo sull'ultimo layer!!!

    this->a_ = new float*[num_of_neurons];
    for(int i=0; i<num_of_neurons; i++)
    	this->a_[i] = new float[num_of_inputs];

    this->delta_ = new float*[num_of_neurons];
    for(int i=0; i<num_of_neurons; i++)
    	this->delta_[i] = new float[num_of_inputs];

    this->bias_ = new float*[num_of_neurons];
    for(int i=0; i<num_of_neurons; i++)
    	this->bias_[i] = new float[num_of_inputs];
}



void FeedForwardLayer::randomInitialize(){
	cout << "Random Initialization...\n";
    //srand() is set at network level!
    for(int neuron=0; neuron < num_of_neurons; neuron++){
    	cout << "--- debug ---\n";
    	//Initialize connection weights
    	cout << " num of input: "<< num_of_inputs << endl;
        for(int input = 0; input < num_of_inputs; input++){
        	cout << "--- inner debug ---\n";
            float r = (float)(rand()%100)/100;
            connections[neuron][input] = r / 3;
            //connections[neuron][input] = 0; //da provate
            //cout << connections[neuron][input] << endl;
        }
        //Initialize bias
        if(l_act_function != act_function::NDIM_SIGMOID)
        	 bias[neuron] = (float)(rand()%100)/300;
        else
			for(int j=0; j< num_of_inputs; j++)
				bias_[neuron][j] = (float)(rand()%100)/300;

    }
}

void FeedForwardLayer::forward(float* x_array){
	  	//x_array is the input array of the layer
	    //std::cout << id << "### Layer forwarding... ###" << std::endl;
	    if(!this->isInputLayer)
	    {
	    	//per ogni neurone i del layer
	        for(int i=0; i<num_of_neurons; i++){
	            a[i] = 0;

	            //per ogni neurone j del layer precedente
	            for(int j=0; j < num_of_inputs; j++){
	                //cout << "a["<< i << "] = "<< a[i] << " + " << connections[i][j] << " * "<< x_array[j]<< endl;
	                a[i] = a[i] + connections[i][j]*x_array[j];
	                a_[i][j] = connections[i][j]*x_array[j] + bias_[i][j];
	            }
	            a[i] = a[i] + bias[i]; //si aggiunge solo il peso. l'output del neurone e' sempre 1
	            out[i] = 0;
	//            for(int i=0; i<num_of_neurons; i++)
	//                cout << " a["<< i <<"] = "<<a[i];
	//            cout << endl;

	            if(l_act_function == act_function::NDIM_SIGMOID){
	            	out[i] = activation(a_[i]);
	            }
	            else
	            	out[i] = activation(a[i]);

	            //cout << "Layer " << this->id << " out["<<i<< "] = "<<out[i]<< endl;
	        }
	//        for(int i=0; i<num_of_neurons; i++)
	//            cout << " a["<< i <<"] = "<<a[i];
	//        cout << endl;
	        //print layer output
	//        for(int i=0; i < num_of_neurons; i++)
	//            cout << out[i] << " ";
	//        cout << endl;
	    }
	    else
	    {
	        out = x_array;
	        //print layer output
	//        for(int i=0; i < num_of_neurons; i++)
	//            cout << out[i] << " ";
	//        cout << endl;
	    }
}

float FeedForwardLayer::backward(){
    //CALLED BY THE OUTPUT LAYER:
    //delta_i = (y^ - y) * f'(a_i)
	float err = 0;
    for(int i=0; i<num_of_neurons; i++){
        //delta[i] = sqrt(error[i]) * act_derivative(a[i]);
    	delta[i] = error[i] * act_derivative(a[i]);
        err = err + error[i]*error[i];
    }
    return err/2;
}

float* FeedForwardLayer::calculate_deltas(){
    //CALLED BY THE OUTPUT LAYER:
    //delta_i = (y^ - y) * f'(a_i)
	float* delta_ = new float[num_of_neurons];
	float err = 0;
    for(int i=0; i<num_of_neurons; i++){
        //delta[i] = sqrt(error[i]) * act_derivative(a[i]);
    	delta[i] = error[i] * act_derivative(a[i]);
    }
    return delta;
}

float* FeedForwardLayer::calculate_deltas(float* errors){
    //CALLED BY THE OUTPUT LAYER:
    //delta_i = (y^ - y) * f'(a_i)
	float* delta_ = new float[num_of_neurons];
	float err = 0;
	//cout << "Output Layer deltas" <<  endl;
    for(int i=0; i<num_of_neurons; i++){
    	delta[i] = errors[i] * act_derivative(a[i]);
    	//cout << i << ")" << delta[i]<<", " ;
    }
    return delta;
}

float FeedForwardLayer::get_error(){
	float err = 0;
	for(int i=0; i<num_of_neurons; i++){
		err = err + error[i]*error[i];
	}
	return err/2;
}

void FeedForwardLayer::backward(Layer2* next_layer/*Subsequent layer*/){
    //Called by a hidden layer:

    //delta_j = sum_i(delta_i * w_ij) * f'(a_j)
    float aux;
    //neurons of the actual layer
    for(int k=0; k<num_of_neurons; k++){
    	//cout << "Neuron "<< k << " ";
        aux = 0;
        for(int j=0; j < next_layer->num_of_neurons; j++){
            //aux = aux + next_layer->get_delta(i) * next_layer->in_weight(i,j);
            aux = aux + next_layer->delta[j] * next_layer->connections[j][k];
        }
        if(l_act_function!= act_function::NDIM_SIGMOID){
        	delta[k] = aux *this->act_derivative(a[k]);
        	//cout << k <<"] "<< delta[k]<<" ";
        }
        else{
        	for(int h=0; h<num_of_inputs; h++){
        		delta_[k][h] = aux *this->act_derivative(a_[k],h);
        		//cout << "["<< k <<", "<<h << "] "<< delta_[k][h] << " ";
        	}
        }
        //cout << endl;
    }
}

void FeedForwardLayer::update_weights(float ni, Layer2* prev_layer){
    if(!isInputLayer){
        //delta_wij = ni * delta_i * o_j
        for(int i=0; i<num_of_neurons; i++){
        	//connection weights update
            for(int j=0; j < num_of_inputs; j++){

            	if(l_act_function!= act_function::NDIM_SIGMOID)
            		//prova
            		connections[i][j] = connections[i][j] - ni*delta[i]*prev_layer->out[j] ;
            	else{
            		connections[i][j] = connections[i][j] - ni*delta_[i][j]*prev_layer->out[j] ;
            		bias_[i][j] = bias_[i][j] - ni*delta_[i][j];
            	}

            }
            //bias weight update
            //w_ib' = w_ib - ni*delta_i*1
            bias[i] = bias[i] - ni*delta[i];

        }
    }
}

float FeedForwardLayer::activation(float x){

    if (this->l_act_function == act_function::SIGMOID){
        return 1/(1 + exp(-x));
    }
    else if(this->l_act_function == act_function::TANH){
        return MlMath::tanh(x);
    }
    else if(this->l_act_function == act_function::LINEAR){
        return x;
    }
    else if(this->l_act_function == act_function::RELU){
        return MlMath::relu(x);
    }
    else
        cout << " No activation function detected!!" << endl;
}


float FeedForwardLayer::activation(float* x){
	return MlMath::ndim_sigmoid(num_of_inputs , x);
}

float FeedForwardLayer::act_derivative(float x){
    if (this->l_act_function == act_function::SIGMOID){
        return MlMath::sigmoid_derivative(x);
    }
    else if(this->l_act_function == act_function::TANH){
        return MlMath::tanh_derivative(x);
    }
    else if(this->l_act_function == act_function::LINEAR){
        return 1;
    }
    else if(this->l_act_function == act_function::RELU){
        return MlMath::relu_derivative(x);
    }
}

float FeedForwardLayer::act_derivative(float *x, int j){
	return MlMath::ndim_sigmoid_partial_der(num_of_inputs, x, j);
}

void FeedForwardLayer::reset_error(){
	for(int i=0; i<num_of_neurons; i++)
		error[i] = 0;
}

float FeedForwardLayer::cumulate_error(float* y_array){
	float err = 0.0;
	for(int i=0; i<num_of_neurons; i++){
		//error[i] =  pow(out[i] - y_array[i], 2);
		error[i] =  out[i] - y_array[i];
		err = err + error[i]*error[i];
	}
	return err/2;
}








