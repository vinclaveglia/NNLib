/*
 * Dataset.h
 *
 *  Created on: 29 giu 2016
 *      Author: vincenzo
 */

#ifndef DATASET_H_
#define DATASET_H_

#include <iostream>
#include <vector>
#include <list>
#include "SequenceItem.h"

using namespace std;

class Dataset {

public:
    Dataset(){};
    Dataset(std::string name);
    Dataset(float** features, float** target, int features_dim, int target_dim, int size);
    void loadFeatures(std::string filepath,  char delimiter);
    void loadTargets(std::string filepath, char delimiter);
    void loadForCharacterLanguageModel(std::string textFile);
    float** features;
    float** target;
    int size;
    int features_dim;
    int target_dim;

    vector< list<SequenceItem> > sequence;
    void load_dataset(std::string filepath, char features_sep, char feature_target_sep);
    void load_dataset(std::string filepath, int features_dim, int target_dim, char delimiter);
    void load_dataset_as_sequence(std::string filepath, char features_sep, char feature_target_sep);

    void print_dataset();

    Dataset* from_to(int start, int end);
    Dataset* from_col_to_col(int start, int end);

    void shuffle();

    Dataset* load_mnist();

private:
    float** load(std::string filepath, int dim);
    vector<vector<float>> load_(std::string filepath, char delimiter);
    float* split(std::string string, char delimiter, int dim);
    vector<float> split_(std::string string, char delimiter);

};

#endif /* DATASET_H_ */
