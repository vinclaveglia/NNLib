/*
 * NeuralNetwork.cpp
 *
 *  Created on: 29 giu 2016
 *      Author: vincenzo
 */

#include "NeuralNetwork.h"
#include <vector>
#include <map>
#include <fstream>
#include <stdlib.h>
#include "FeedForward/FeedForwardLayer.h"

#include "act_function.h"

using namespace std;

int argmax(float* array, int size){
	int max_ind = -1;
	float max_val = -1000;
	for(int i=0; i<size; i++){
		if(array[i] > max_val){
			max_ind = i;
			max_val = array[i];
		}
	}
	return max_ind;
}

void NeuralNetwork::print_vect(float* v, int size){
	for(int i=0; i<size; i++){
		cout << v[i] << " ";
	}
	cout << endl;
}

Layer2* NeuralNetwork::get_output_layer(){
	return layer_list.at( layer_list.size()-1 );
}

float* NeuralNetwork::calculate_deltas(){
	//output_layer
	Layer2* output_layer;
	output_layer = layer_list.at( layer_list.size()-1 );
    return output_layer->calculate_deltas();
}

float* NeuralNetwork::calculate_deltas(float* errors){
	//output_layer
	Layer2* output_layer;
	output_layer = layer_list.at( layer_list.size()-1 );
    return output_layer->calculate_deltas(errors);
}

float NeuralNetwork::train_step(Dataset* dataset){
	float error, step_error = 0;
	Layer2* ol  = get_output_layer();

    for(int i=0; i<dataset->size; i++){

        forward( dataset->features[i]);

        error = cumulate_error( dataset->target[i]);

        //calculate_deltas();

        //calcola i delta sull'output layer
        calculate_deltas(ol->error);

        //retropropaga
        backward2();

        //aggiorna i pesi
        update_weights();

        step_error = step_error + error;
    }

    return step_error;
}



float NeuralNetwork::network_error_tmp(Dataset* dataset){
	float error = 0, pattern_error = 0;

	for(int i=0; i < dataset->size; i++){

		forward( dataset->features[i]);

		//print_vect(dataset->features[i], dataset->features_dim);
		//print_vect(dataset->target[i], dataset->target_dim);
		//this->output();

		pattern_error = cumulate_error( dataset->target[i]);
		//cout << "--- pattern error: "<< pattern_error << endl;

	}
	error = error + pattern_error;

	return error;
}



void NeuralNetwork::train2(Dataset* dataset){
    vector<float> o_error;
    float epoch_error;
    float tmp_error;
    cout << "Starting training v2\n";

    for(int epoch=0; epoch < epochs; epoch++){
        epoch_error = 0;
        cout << "=== Epoch "<< epoch << "===\n";

        epoch_error = train_step(dataset);

        cout << ", ERROR = "<< epoch_error << "====\n";
        o_error.push_back(epoch_error);
    }
    cout << "\n===== Trained with train 2 =====\n";

    ofstream err_file;
    err_file.open("net_error.txt");
    for(int i=0; i < o_error.size(); i++)
        err_file <<i <<" "<<o_error.at(i)<<endl;
    err_file.close();
}

void NeuralNetwork::train2(Dataset* dataset, Dataset* validation_set){
    vector<float> train_errors;
    vector<float> val_errors;
    float train_error;
    float tmp_error;
    float val_error = 0;
    cout << "Starting training v2 ...\n";

    /*
    for(int epoch=0; epoch < epochs; epoch++){
        train_error = 0;
        cout << "=== Epoch "<< epoch;

        train_error = train_step(dataset) / dataset->size;

        val_error = network_error_tmp(dataset) / dataset->size ;

        cout << ", TRAIN ERROR = "<< train_error << ", VALIDATION ERROR: "<< val_error<< "====\n";
        train_errors.push_back(train_error);
        val_errors.push_back(val_error);
    }*/

    int step_error;
    for(int epoch=0; epoch < epochs; epoch++){
    	cout << "=== Epoch "<< epoch << " ==="<<endl;
    	//training step
    	float tr_error = 0;
    	float error=0;
    	val_error = 0;

    	//Training Step
    	for(int i=0; i<dataset->size; i++){
    	//for(int i=0; i<1; i++){
    		forward( dataset->features[i]);


			error = cumulate_error( dataset->target[i]);
			//calculate_deltas();
			//calcola i delta sull'output layer
			calculate_deltas(get_output_layer()->error);
			//retropropaga

			backward2();

			//aggiorna i pesi

			update_weights();
			tr_error = tr_error + error;
    	}
    	tr_error = tr_error / dataset->size;

    	//Validation Step
    	error = 0;
    	/*for(int i=0; i<validation_set->size; i++){
    		forward( validation_set->features[i]);
			error = cumulate_error( validation_set->target[i]);
			val_error = val_error + error;
    	}*/
    	val_error = val_error / validation_set->size;
    	train_errors.push_back(tr_error);
    	val_errors.push_back(val_error);
    }

    cout << "===== Trained with train 2 =====\n";

    ofstream train_err_file, val_err_file;
    train_err_file.open("training_error.txt");
    val_err_file.open("validation_error.txt");

    for(int i=0; i < train_errors.size(); i++){
        train_err_file <<	i <<" "<<train_errors.at(i)<<endl;
        val_err_file << i << " "<< val_errors.at(i) << endl;
    }
    train_err_file.close();
    val_err_file.close();
    cout << "End training v2 ...\n";
}

void NeuralNetwork::train(Dataset* dataset){
    vector<float> o_error;
    float epoch_error;
    float tmp_error;
    cout << "Starting training...\n";

    for(int epoch=0; epoch < epochs; epoch++){
        epoch_error = 0;
        cout << "=== Epoch "<< epoch << " ===";
        for(int i=0; i<dataset->size; i++){
        	reset_error();
            forward( dataset->features[i]);
            cumulate_error( dataset->target[i]);
            tmp_error = backward();
            //store_deltas(i);
            update_weights();
            epoch_error = epoch_error + tmp_error;
        }
        //dividiamo epoch error per il numero dei record? no?
        //epoch_error = epoch_error / dataset->size;
        cout << ", ERROR = "<< epoch_error << "====\n";
        o_error.push_back(epoch_error);
    }

    ofstream err_file;
    err_file.open("net_error.txt");
    for(int i=0; i < o_error.size(); i++)
        err_file <<i <<" "<<o_error.at(i)<<endl;
    err_file.close();
}

void print_array(float* array, int size){
	for(int i=0; i< size; i++)
		cout << array[i] << " ";
	cout << endl;
}

void print_array_on_file(float* array, int size, ofstream file){
	for(int i=0; i< size; i++)
		file << array[i] << " ";
}

float NeuralNetwork::scoreClassifier(Dataset* dataset){
	float correct = 0;
	float wrong = 0;
	float score = 0;
	//prendo l'output layer
	Layer2* out_l = layer_list.at( layer_list.size()-1 );

	//if(dataset->target_dim != out_l->size())

	for(int i=0; i<dataset->size; i++){

		forward( dataset->features[i]);
		//cout << "Target dim is: "<< dataset->target_dim << "!!!\n";
		//cout << "Output layer dim is: "<< out_l->size()<< "!!!\n";

		/*for(int j=0; j < out_l->size(); j++){
			cout << out_l->out[j] << " ";
		}
		cout << " -> ";
		for(int j=0; j <dataset->target_dim; j++){
			cout << dataset->target[i][j] << " ";
		}*/
		unsigned int target_class = argmax(dataset->target[i], dataset->target_dim);
		unsigned int predicted_class = argmax(out_l->out, out_l->size());

		if (target_class == predicted_class)
			correct++;
		else
			wrong++;

		//cout << "argmaxS: "<< target_class << " "<< predicted_class;
		//cout << endl;
	}
	score = correct / (correct + wrong);
	cout << "Classifier SCORE is: "<<score<<"!!!\n";
}


float NeuralNetwork::score(Dataset* dataset){
	float correct = 0;
	float wrong = 0;
	float score = 0;
	//prendo l'output layer
	Layer2* out_l = layer_list.at( layer_list.size()-1 );

	//crea 2 file, uno per y e uno per y^.
	ofstream output_file, target_file;
	output_file.open("outputs.txt");
	target_file.open("targets.txt");


	for(int i=0; i<dataset->size; i++){

		forward( dataset->features[i]);
		//iterate on output neurons

		//print_array(dataset->features[i], 1);

		//print_array_on_file(dataset->features[i],1, output_file);
		//print_array_on_file(out_l->out ,1, output_file);

		output_file << dataset->features[i][0] << " " << out_l->out[0] << endl;

		for(int j=0; j<out_l->size(); j++){
			cout << out_l->out[j] << " - "<< dataset->target[i][j] << endl;
			if(out_l->out[j] == dataset->target[i][j])
				correct++;
			else
				wrong++;
		}
	}
	output_file.close();

	score = correct/(correct+wrong);
	cout << "-----------------------\n";
	cout << "Network classification score is "<< score << "%\n";
	return score;
}


void NeuralNetwork::train(float** dataset){
	cout << "Training not implementes yet" << endl;
}

void NeuralNetwork::initialize(){
    cout << "Initializing network..." << endl;
    for(Layer2* layer : layer_list){
    	if(!layer->isInputLayer){
    		cout << "Initializing layer "<<layer->id<< " ...\n";
			layer->randomInitialize();
			cout << "Layer "<<layer->id<< " initialized\n";
    	}
    }
}

void NeuralNetwork::setInputLayer(int num_of_neurons){
    Layer2* inputLayer = new FeedForwardLayer(num_of_neurons);
    layer_list.clear();
    layer_list.push_back(inputLayer);
    cout << "Input layer added!" << endl;
}

void NeuralNetwork::addLayer(int num_of_neurons, string act_function){
    //take num of neurons of previous layer
    int prevLayerIndex = layer_list.size() - 1;
    int layerId = layer_list.size();

    int prevLayerSize = layer_list.at(prevLayerIndex)->size();

    cout << "Prev size = " << prevLayerSize << endl;
    Layer2* layer = new FeedForwardLayer(num_of_neurons, prevLayerSize, layerId, act_function); //avrà la matrice dei pesi
    layer_list.push_back(layer);
    cout << "Added layer with "<< num_of_neurons << "neurons!\n";

}



float* NeuralNetwork::output(){
    //int last_layer_i = layer_list.size() - 1;
    //Layer2* l =layer_list.at(last_layer_i);

    Layer2* l = this->get_output_layer();
    //cout << "net output: ";
    //for(int i=0; i<l->num_of_neurons; i++){
    //	cout << l->out[i] << " ";
    //}
    //cout << endl;
    return l->output();
}

void NeuralNetwork::forward(float* x /*Input vector*/){

    for(int i=0; i< layer_list.size(); i++){
        Layer2* layer = layer_list.at(i);

        if(i!=0){
            //layer->printWMatrix();
            Layer2* prev_layer = layer_list.at(i-1);
            layer->forward( prev_layer->output());
        }
        else{
            //is the input layer
            layer->forward(x);
        }
    }
    //return output
}

float NeuralNetwork::backward(){
    Layer2* actual;
    Layer2* subsequent;
    float error;

    //Get output layer
    actual = layer_list.at( layer_list.size()-1 );
    error = actual->backward(); //this have to be applied only to output layer

    //Get the hidden layers. NB: The input layer has not to be backwarded!! (i > 0)
    for(int i = layer_list.size()-2; i > 0; i-- ){
        actual = layer_list.at(i);
        subsequent = layer_list.at(i+1);
        actual->backward( subsequent );
    }
    return error;
}

void NeuralNetwork::backward2(){
    Layer2* actual;
    Layer2* subsequent;
    float error;

    //Get output layer
    actual = layer_list.at( layer_list.size()-1 );

    //Get the hidden layers. NB: The input layer has not to be backwarded!! (i > 0)
    for(int i = layer_list.size()-2; i > 0; i-- ){
        actual = layer_list.at(i);
        subsequent = layer_list.at(i+1);
        actual->backward( subsequent );
    }
}

float NeuralNetwork::output_error(){
	Layer2* output_layer = get_output_layer();
	return output_layer->get_error();
}



void NeuralNetwork::setLearningRate(float ni){
    this->learning_rate = ni;
    cout << "Learning rate = "<< learning_rate << endl;
}

void NeuralNetwork::setEpochs(int noe){
    this->epochs = noe;
    cout << "Number of epochs = "<< epochs << endl;
}

void NeuralNetwork::update_weights(){
    Layer2* actual;
    Layer2* previous;
    for(int i=layer_list.size()-1; i>0; i--){
        actual = layer_list.at(i);
        previous = layer_list.at(i-1);
        actual->update_weights(learning_rate, previous);
    }
}

void NeuralNetwork::reset_error(){
	//take output layer
	Layer2* layer = layer_list.at( layer_list.size()-1 );
	layer->reset_error();
}

float NeuralNetwork::cumulate_error(float* y_array){
	//agisce solo sul layer di output
	//Layer2* layer = layer_list.at( layer_list.size()-1 );
	Layer2* layer = this->get_output_layer();
	return layer->cumulate_error( y_array);
}


void NeuralNetwork::store_deltas(int dataset_pos){
	for(Layer2* layer: layer_list){
		cout << "layer "<< layer->id <<" storing\n";

		for(int i=0; i<layer->num_of_neurons; i++){
			cout <<"neuron "<<i<<" pos " << dataset_pos <<endl;
			cout << "delta = " << layer->delta[i]<<endl;

			layer->delta_t[i][dataset_pos] = layer->delta[i];
		}
	}
}







