/*
 * FeedForwordLayer.h
 *
 *  Created on: 29 giu 2016
 *      Author: vincenzo
 */

#ifndef FEEDFORWARDLAYER_H_
#define FEEDFORWARDLAYER_H_

#include "Layer2.h"


class FeedForwardLayer : public Layer2{
public:

    FeedForwardLayer(int num_of_neurons);

    FeedForwardLayer(int num_of_neurons, int previous_layer_size, int layer_id, std::string act_function);

    void randomInitialize() override; //to do

    void forward(float* x_array) override; //to do

    float backward() override; //to do

    void backward(Layer2* layer/*Subsequent layer*/) override; //to do

    void update_weights(float learning_rate, Layer2* previous_layer) override; //to do

    void reset_error() override;

    float cumulate_error(float* y_array) override;

    void reset() {};

    float get_error() override;

    float* calculate_deltas() override;
    float* calculate_deltas(float* errors) override;


private:
    float activation(float x);
    float activation(float* x);
    float act_derivative(float x);
    float act_derivative(float *x, int j);
};

#endif /* FEEDFORWARDLAYER_H_ */
