#ifndef LAYER2_H
#define LAYER2_H
#include <iostream>


class Layer2{
    //non-virtual members of derived classes cannot be accessed through a reference of the base class
public:
    Layer2(){};

    Layer2(int num_of_neurons);

    //da sovrascrivere?? tanto ognuno avrà il suo --> da rivedere
    //Layer2(int num_of_neurons, int previous_layer_size, int id_layer, std::string act_function);

    //strettamente legato alla istanza di layer
    virtual void randomInitialize(float scale_factor)=0;

    //strettamente legato alla istanza di layer
    virtual void forward(float* x_array) = 0;

    //strettamente legato alla istanza di layer
    virtual float backward() = 0;

    //strettamente legato alla istanza di layer
    virtual void backward(Layer2* layer/*Subsequent layer*/) = 0;

    //strettamente legato alla istanza di layer, dipende dalla struttura delle connessioni
    virtual void update_weights(float learning_rate, Layer2* previous_layer) = 0;

    virtual void reset_error() = 0;

    virtual void reset() = 0;

    virtual float cumulate_error(float* y_array) = 0;

    //opzionali
    virtual float get_error() = 0;

    virtual float* calculate_deltas() = 0;
    virtual float* calculate_deltas(float* error) = 0;

    void printWMatrix();

    //implemented here
    float* output();
    int size();
    int getId();

    //////////////////////////////////

    int id;
    int num_of_neurons;
    float** connections; //dimensioni: (#neurons x #inputs))
    float* delta; //delta parameter used in calculation of weight updates
    float** delta_; // delta_ arrays for multidimensional sigmoid functions, one element for each connection

    int num_of_inputs = 0; //num of neuron of the previous layer
    bool isInputLayer = false;
    bool isOutputLayer = false;
    std::string l_act_function; //layer activation function
    float* bias; //un bias weight per neurone
    float* out; //the output of the neuron after the activation function
    float* a; // the result of the moltiplication w*x

    float** a_; // a_i arrays for multidimensional sigmoid functions
    float** bias_; // a bias arrays for each neuron [multidimensional sigmoid functions]

    float* error; //the output error (used in output layers for batch learning)
    float** delta_t; //experiment
    std::string type;
    std::string FEEDFORWARD = "FeedForward";
    std::string RECURRENT = "Recurrent";
};

#endif /* LAYER2_H */
