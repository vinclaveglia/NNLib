
#include "Layer2.h"

#include <iostream>

//dovra essere l'oggetto layer del futuro...
using namespace std;

Layer2::Layer2(int num_of_neurons){
    this->id = 0;
    this->num_of_neurons = num_of_neurons;
    this->isInputLayer = true;
    //è un input layer, non ha matrice dei pesi
    //l'output del layer corrisponde al suo input
}

float* Layer2::output(){
    return this->out;
}

int Layer2::getId(){
    return this->id;
}

int Layer2::size(){
    return num_of_neurons;
}
