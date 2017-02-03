/*
 * Dataset.cpp
 *
 *  Created on: 29 giu 2016
 *      Author: vincenzo
 */

#include <fstream>
#include <regex>
#include <iostream>
#include <list>
#include <vector>
#include <cassert>
#include <exception>
#include <sstream>
#include "Dataset.h"


using namespace std;

float* ssplit(string line, char delimiter, int dim){
	istringstream stringstream(line);
	string token;
	std::vector<float> fv;
	float* array = new float[dim];
	int i=0;

	while(std::getline(stringstream, token, delimiter)){
		float f = std::stof(token);
		fv.push_back(f);
		array[i] = f;
		i++;
	}
	//return fv.data(); //return a float array
	return array;
}

float** vector_of_vector_to_array(vector<vector<float>> matrix){
    //[temporary] copy of vector<vector<float>> to float matrix
    float** array = new float*[matrix.size()];
    int dim = matrix[0].size();
    for(int i=0; i<matrix.size(); i++){
    	array[i] = new float[dim];
    	copy(matrix[i].begin(), matrix[i].end(), array[i]);
    }
    return array;
}

int get_dimension(string line, char delimiter){
	istringstream stringstream(line);
	string token;
	int dim = 0;
	std::vector<float> fv;

	while(std::getline(stringstream, token, delimiter)){
		float f = std::stof(token);
		fv.push_back(f);
	}
	return fv.size(); //return a float array
}

void print_array2(float* array, int size){
	for(int i=0; i< size; i++)
		cout << array[i] << " ";
	cout << endl;
}

Dataset::Dataset(float** features, float** target, int features_dim, int target_dim, int size){
	this->features = features;
	this->features_dim = features_dim;
	this->target = target;
	this->target_dim = target_dim;
	this->size = size;
	cout << "Dataset created\n";
}

void Dataset::load_dataset(std::string filepath, int features_dim, int target_dim, char delimiter){
	//load the whole dataset in memory
	vector<vector<float>> dataset = this->load_(filepath, delimiter);
	vector<vector<float>> features;
	vector<vector<float>> target;

	for(vector<float> row: dataset){

		vector<float> tmp_f (row.begin(), row.begin()+features_dim);

		features.push_back(tmp_f);

		vector<float> tmp_t (row.begin()+features_dim, row.end());

		target.push_back(tmp_t);
	}

	this->features_dim = features[0].size();
	this->target_dim = target[0].size();

	this->features = vector_of_vector_to_array(features);
	this->target = vector_of_vector_to_array(target);
}

void Dataset::load_dataset(std::string filepath, char features_sep, char feature_target_sep){
	string line;
    ifstream is(filepath);
    int counter = 0;
    while(is >> line){
        counter++;
    }
    this->size = counter;
    is.close();
    float** matrix = new float*[counter];
    float** matrix_target = new float*[counter];

    is.open(filepath);
    int i = 0;
    int pos;
	std::string target;
	std::string features;

	counter=0;
    while(is >> line){
    	//trova la posizione di feature_target_sep
    	//cout <<"-- "<<line <<endl;
    	pos = line.find(feature_target_sep);
    	target = line.substr(pos+1);
    	features = line.substr(0, pos);
    	//cout << "feature: " << features << " target: " << target << endl;

    	matrix[i] = ssplit(features, features_sep, 1);
    	matrix_target[i]  = ssplit(target, features_sep, 1);

    	//print_array2(matrix[i], 1);
    	//print_array2(matrix_target[i], 1);

    	i++;
    	if(counter == 0){
    		features_dim = get_dimension(features, features_sep);
    		target_dim = get_dimension(target, features_sep);
    		counter=1;
    	}

    }
    this->features = matrix;
    this->target = matrix_target;
}

void Dataset::load_dataset_as_sequence(std::string filepath, char features_sep, char feature_target_sep){

	SequenceItem item; //is the pair <features_array, target_array>
	list<SequenceItem> msequence; //is the whole dataset

	string line;
	ifstream is(filepath); //open the stream

	int pos;
	int counter = 0;
	std::string target;
	std::string features;

	while(is >> line){
    	pos = line.find(feature_target_sep);
    	target = line.substr(pos+1);
    	features = line.substr(0, pos);
    	if(counter == 0){
    		features_dim = get_dimension(features, features_sep);
    		target_dim = get_dimension(target, features_sep);
    		counter=1;
    	}
    	item.features = ssplit(features, features_sep, features_dim);
    	item.target  = ssplit(target, features_sep, target_dim);

    	msequence.push_back(item);

	}
	sequence.clear();
	sequence.push_back(msequence);


}

void Dataset::print_dataset(){
	cout << "---------------------------------\n";
	cout << "-           DATASET             -\n";
	cout << "---------------------------------\n";
	cout << "Size: "<< size << ", Features dim: "<< features_dim << ", Target dim: "<< target_dim << endl;
	cout << "---------------------------------\n";
	for(int i=0; i<size; i++){
		for(int j=0; j<features_dim; j++){
			cout << features[i][j] << " ";
		}
		cout << " - ";
		for(int j=0; j<target_dim; j++){
			cout << target[i][j] << " ";
		}
		cout << endl;
	}
	cout << "---------------------------------\n";
}

void print_vect(float* v, int size){
	for(int i=0; i<size; i++){
		cout << v[i] << " ";
	}
	cout << endl;
}


float** Dataset::load(string filepath, int dim){

    string line;
    ifstream is(filepath);
    int counter=0;
    //if(is.is_open())
    while(is >> line){
        counter++;
        //cout << line << "_" << counter<< endl;
    }
    this->size = counter;
    is.close();
    //alloca matrice
    float** matrix = new float*[counter];
    float* row;
    //riapro uno stream chiuso
    is.open(filepath);
    int i  = 0;
    while(is >> line){
       row = split(line,',', dim);
       matrix[i] = row;//row è un array
       //stampa l'array
       //print_vect(matrix[i],4);
       i++;
    }
    return matrix;
}

vector<vector<float>> Dataset::load_(string filepath, char delimiter){
    vector<vector<float>> matrix;
    vector<float> row;

	cout << "Loading data...\n";
    string line;
    ifstream is(filepath);
    bool header = false;

    while(getline(is, line)){
       if(header==false){
    	   //add exception in reading line as composed of float (it could be a header!)
    	   row = split_(line, delimiter);
    	   matrix.push_back(row);
       }
       else{
    	   cout << "Header: " << line <<  endl;
    	   header = false;
       }
    }
    this->size = matrix.size();
    cout << "-- Data at " << filepath << " loaded --\n";
    return matrix;
}



void Dataset::loadFeatures(string filepath, char delimiter){
	//this->features_dim = dim;
    //this->features = this->load(filepath, dim);
	vector<vector<float>> mdata = this->load_(filepath, delimiter);
	this->features = vector_of_vector_to_array(mdata);
}

void Dataset::loadTargets(string filepath, char delimiter){
	//this->target_dim = dim;
    //this->target = this->load(filepath, dim);
	vector<vector<float>> mdata = this->load_(filepath, delimiter);
	this->target = vector_of_vector_to_array(mdata);
}

float* Dataset::split(string line, char delimiter, int dim){
    string token;
    istringstream stringstream(line);
    float* array = new float[4];// <-- OCCHIO!!!
    int col_counter = 0;
    float f;
    while(std::getline(stringstream, token, delimiter)) {
        if(col_counter<dim){
            f = std::stof(token);
            array[col_counter] = f;
            col_counter++;
            //cout << f << endl;
        }
    }
    //cout << "----------------------------------------\n";
    return array;
}

vector<float> Dataset::split_(string line, char delimiter){
    string token;
    istringstream stringstream(line);
    vector<float> row;
    float f;

    while(std::getline(stringstream, token, delimiter)) {
    	try{
    		f = std::stof(token);
    		row.push_back(f);
    	}catch(exception& e){
    		//cout << "Standard exception: " << e.what() << endl;
    	}
    }

    return row;
}

Dataset* Dataset::from_to(int start, int end){
	assert(end >= start);
	assert(end <= this->size);
	int msize = end - start;
	cout << "New dataset size is "<< msize<< endl;
	float** mfeatures = new float*[msize];
	float** mtarget = new float*[msize];

	for(int i=0; i< msize; i++){
		mfeatures[i] = new float[features_dim];
		mtarget[i] = new float[target_dim];
	}
	cout << "Space allocated" << endl;
	for(int i=0; i< msize; i++){
		mfeatures[i] = features[i+start];
		mtarget[i] = target[i+start];
	}

	Dataset* newdataset =
			new Dataset(mfeatures, mtarget, features_dim, target_dim, msize);

	return newdataset;
}

void Dataset::shuffle(){
	std::vector<int> myvector;

	for (int i = 0; i< size; ++i)
		myvector.push_back(i);

	 std::random_shuffle ( myvector.begin(), myvector.end() );

	 float** features2 = new float*[size];
	 float** target2 = new float*[size];

	 for(int i=0; i<size; i++){
		 features2[i] = new float[features_dim];
		 target2[i] = new float[target_dim];
	 }

	 for(int i=0; i<size; i++){
		 //cout << i << " -->" << myvector.at(i) << endl;
		 features2[i] = this->features[myvector.at(i)];
		 target2[i] = this->target[myvector.at(i)];
	 }
	 features = features2;
	 target = target2;

	 //for(int i=0; i< myvector.size(); i++){
	 // cout << myvector.at(i) << endl;
	 //}


}

void split_words(string line, char delimiter){
    string token;
    istringstream stringstream(line);
    float* array = new float[4];
    int col_counter = 0;
    float f;
    cout << "stiamo provando\n";
    while(std::getline(stringstream, token, delimiter)) {
    	cout << "Token = "<<token<<endl;
    }
}

float* char_to_x(char c){
	c = toupper(c);

	std::string alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

	float * x = new float[alphabet.size()];

	for(int i=0; i<alphabet.size(); i++)
		x[i] = 0;

	int pos = alphabet.find(c);
	if(pos!=string::npos){
		x[pos] = 1;
	}
	/*cout << "Char: "<< c;
	for(int i=0; i<alphabet.size(); i++)
		cout << " " << x[i] << " ";
	cout << endl;*/
	return x;
}



void Dataset::loadForCharacterLanguageModel(string textFile){
	/*Given a text file, code every characte as a one-hot alphabet lenght array
	 * Set as target the next character in the word*/

	//read the text file
	ifstream is(textFile);
	string line;

	string token;
	char delimiter=' ';
	float * x;
	SequenceItem item;

	list<SequenceItem> char_list; //is the word (a sequence of character)

	sequence.clear();

	while(is >> line){

		istringstream stringstream(line);

	    while(std::getline(stringstream, token, delimiter)) {
	    	cout << "Token = "<<token<<endl;
	    	char_list.clear();

	    	//scanning lettera per lettera ATTENZIONE CHE COSÌ NON PRENDE LE LETTERE SINGOLE
	    	for(int i=0; i<token.size()-1 ; i++){
	    		//x = char_to_x(token[i]);
	    		item.features = char_to_x(token[i]);
	    		item.target = char_to_x(token[i+1]);

	    		char_list.push_back(item);

	    	}
	    	//salva la sequenza di array;
	    	//word_list.push_back(char_list);
	    	sequence.push_back(char_list);
	    }
	}

	//string line = "QUESTA E SOLO UNA PROVA AE EH ZAE QUESTA E SOLO UNA PROVA EH ZA";

	cout << line  << endl;
    //line.erase(std::remove( line.begin(), line.end(), '\n'), line.end());
}



int ReverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return((int) ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}


void read_Mnist(string filename, vector<vector<double> > &vec){

	ifstream file (filename, ios::binary);
	cout << "doing............\n";
	if (file.is_open()){
		cout << "doing...\n";
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;

		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);

		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		file.read((char*) &n_rows, sizeof(n_rows));
		n_rows = ReverseInt(n_rows);

		file.read((char*) &n_cols, sizeof(n_cols));
		n_cols = ReverseInt(n_cols);

		for(int i = 0; i < number_of_images; ++i)
		{
			vector<double> tp;
			for(int r = 0; r < n_rows; ++r)
			{
				for(int c = 0; c < n_cols; ++c)
				{
					unsigned char temp = 0;
					file.read((char*) &temp, sizeof(temp));

					tp.push_back((double)temp);
				}
			}
			vec.push_back(tp);
		}
	}
	else{
		//cerr (filename);
	}
}


void read_Mnist_Label(string filename, vector<double> &vec){

	ifstream file (filename, ios::binary);

	if (file.is_open()){
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*) &magic_number, sizeof(magic_number));
		magic_number = ReverseInt(magic_number);
		file.read((char*) &number_of_images,sizeof(number_of_images));
		number_of_images = ReverseInt(number_of_images);

		for(int i = 0; i < number_of_images; ++i){
			unsigned char temp = 0;
			file.read((char*) &temp, sizeof(temp));
			vec[i]= (double)temp;
		}
	}
}



Dataset* Dataset::load_mnist(){
	Dataset* d;

	string images = "/home/vincenzo/workspace_java/NNLib/MNIST/t10k-images-idx3-ubyte";
	string labels = "/home/vincenzo/workspace_java/NNLib/MNIST/t10k-labels-idx1-ubyte";

	int number_of_images = 10000;
	int image_size = 28 * 28;


	//read MNIST iamge into double vector
	vector<vector<double> > vec;

	read_Mnist(images, vec);

	cout<<vec.size()<<endl;
	cout << "------- debug --------\n";
	cout<<vec[0].size()<<endl;
	cout << "------- debug --------\n";

	//read MNIST label into double vector
	vector<double> vec_l(number_of_images);
	read_Mnist_Label(labels, vec_l);
	cout<<vec.size()<<endl;
	cout << "done?\n";
	return d;
}













