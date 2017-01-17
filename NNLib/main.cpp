/*
 * main.cpp
 *
 *  Created on: 29 giu 2016
 *      Author: vincenzo
 */

#include <cstdlib>
#include <iostream>
#include <fstream>

#include "Dataset/Dataset.h"
#include "NeuralNetwork.h"
#include "act_function.h"

using namespace std;



void testMLP(){
    //Dataset* dataset = new Dataset("data");
    Dataset dataset;

    Dataset mnist;
    //mnist.load_mnist();

    dataset.loadFeatures("/home/vincenzo/Scrivania/Data/iris.data",4);
    dataset.loadTargets("/home/vincenzo/Scrivania/Data/iris.target",3);
    //dataset.print_dataset();


    dataset.shuffle();

    Dataset* training = dataset.from_to(0,90);
    Dataset* validation = dataset.from_to(90,120);
    Dataset* test = dataset.from_to(120,150);

    training->print_dataset();
    test->print_dataset();

//    dataset.loadFeatures("/home/vincenzo/Scrivania/Data/xor.data",2);
//    dataset.loadTargets("/home/vincenzo/Scrivania/Data/xor.target");

    NeuralNetwork rete;
    rete.setInputLayer(4);                      //Input layer

    //rete.addLayer(5, act_function::NDIM_SIGMOID);
    rete.addLayer(5, act_function::SIGMOID);

    rete.addLayer(3, act_function::SIGMOID);	//output layer

    rete.setLearningRate(0.08);

    rete.setEpochs(2000);

    cout << "Initializing...\n";

    rete.initialize();

    rete.train2(training, validation);

    rete.scoreClassifier(&dataset);

    cout << "Finish" << endl;
}

void test_simple_mlp(){
	Dataset dataset;
	string filepath = "/home/vincenzo/workspace_java/RNN/data_generation/sin_data"; //this is noisy

	Dataset real_dataset;

	char feature_separator = ',';
	char feature_target_separator = ',';
	dataset.load_dataset(filepath,feature_separator,feature_target_separator);

	NeuralNetwork mlp;
	mlp.setInputLayer(5);
	mlp.addLayer(20, act_function::SIGMOID);
	mlp.addLayer(1, act_function::SIGMOID);

	mlp.setLearningRate(0.01);
	mlp.setEpochs(1000);
	mlp.initialize();
	mlp.train(&dataset);

	ofstream output_file;
	ofstream target_file;

	output_file.open("outputs_mlp.txt");
	target_file.open("targets_mlp.txt");

	for(int i=0; i<dataset.size; i++){
		mlp.forward(dataset.features[i]);
		mlp.output();
		cout << "wrinting..." << i << " "<< mlp.output()[0] << endl;
		output_file << i << " "<< mlp.output()[0] << endl;
		target_file << i << " "<< dataset.target[i][0] << endl;
	}
	output_file.close();
	target_file.close();
}

void vae(){

	Dataset dataset; //da creare

	int X_DIM = 4, Z_DIM = 2;

	NeuralNetwork encoder, decoder;

	encoder.setInputLayer(X_DIM);
	encoder.addLayer(10, act_function::SIGMOID);
	encoder.addLayer(Z_DIM*2, act_function::LINEAR);

	decoder.setInputLayer(Z_DIM);
	decoder.addLayer(10,act_function::SIGMOID);
	decoder.addLayer(X_DIM, act_function::LINEAR);

	//train
	//setta la particolare funzione di loss per il decoder
	float *pattern;
	float *z, *z_tilde;
	for(int epoch=0; epoch<500; epoch++){
		for(int i=0; i<dataset.size; i++){
			pattern = dataset.features[i];

			encoder.forward(pattern);
			//mu = encoder.output[mu]
			//sigma = encoder.output[sigma]
			//z = random sample
			//z_tilde = z * sigma + mu
			decoder.forward(z_tilde);
			//calcola loss

			//retropropaga
		}
	}
}

void testRegression(){
    Dataset dataset;
    Dataset dataset_to_test;

    char SEP = ' ';
    int FEATURE_DIM = 1;
    int TARGET_DIM = 1;

    dataset.load_dataset("./Data/x_y.txt" , FEATURE_DIM,  TARGET_DIM, SEP);
    dataset_to_test.load_dataset("./Data/x_y.txt" , FEATURE_DIM,  TARGET_DIM, SEP);


    //dataset.print_dataset();
    int perc = dataset.size / 10;

    //trainin percentage
    float TP = 0.8;

    dataset.shuffle();

    Dataset *training = dataset.from_to(0, dataset.size*0.8);
    Dataset *validation = dataset.from_to(dataset.size*0.8, dataset.size*0.9);
    Dataset *testing = dataset.from_to(dataset.size*0.9, dataset.size);

    NeuralNetwork rete;
    rete.setInputLayer(1);
    rete.addLayer(30, act_function::SIGMOID);
    //pushed branch2
    rete.addLayer(1, act_function::LINEAR);

    rete.setLearningRate(0.002);
    rete.setEpochs(2000);
    rete.initialize();

    rete.train2(training, validation);

    ofstream predictions_file;
    predictions_file.open("./Data/predictions.txt");
    float *y, *x;

    ///////////////////////////////////////////////////

    for(int i=0; i<dataset_to_test.size; i++){
    	x = dataset_to_test.features[i];
    	rete.forward( x );
    	y = rete.output();
    	//cout << x[0]<< " "<< y[0] << endl;
    	predictions_file << x[0]<< " "<< y[0] << endl;
    }
    predictions_file.close();
    cout << "Finish" << endl;
}

void testDatasetShuffle(){
    Dataset dataset, *dataset2;


    char SEP = ' ';
    int FEATURE_DIM = 1;
    int TARGET_DIM = 2;

    dataset.load_dataset("./Data/x_y.txt" , FEATURE_DIM,  TARGET_DIM, SEP);
    dataset2 = dataset.from_to(0, 10);
    dataset2->print_dataset();
    dataset2->shuffle();
    dataset2->print_dataset();

}

int main() {
	srand(1);
    //testMLP();
	//test_simple_mlp();
	testRegression();
	//testDatasetShuffle();

    cout << "===== This is NNLib! =====" << endl;
    return 0;
}



