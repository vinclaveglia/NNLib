/*
 * NeuralNetwork.h
 *
 *  Created on: 29 giu 2016
 *      Author: vincenzo
 */

#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_

#include "Dataset/Dataset.h"
#include <vector>
#include "FeedForward/Layer2.h"

using namespace std;

class NeuralNetwork {

public:
	NeuralNetwork(){};
    string name;
    void train(float** d); //float matrix pointer
    void train(Dataset* dataset);

    void train2(Dataset* training_set, Dataset* validation_set);

    void train2(Dataset* training_set);


    float train_step(Dataset* dataset);

    float* calculate_deltas();
    float* calculate_deltas(float* errors);

    float output_error();

    void initialize();
    void setInputLayer(int num_of_neurons);

    void addLayer(int num_of_neurons, string act_function);

    float network_error_tmp(Dataset* dataset);

    float* output();

    void forward(float*); //a float array (the input x))

    float backward();

    void backward2();

    void setLearningRate(float ni);

    float learning_rate = 0.01;

    void update_weights();

    void setEpochs(int noe);

    float cumulate_error(float* y_array);

    void reset_error();

    Layer2* get_output_layer();

    float score(Dataset* dataset);

    float scoreClassifier(Dataset* dataset);

    void store_deltas(int dataset_position);

    void print_vect(float* v, int size);


private:
    int epochs;
    float network_error;
    vector<Layer2*> layer_list;
};

#endif /* NEURALNETWORK_H_ */
