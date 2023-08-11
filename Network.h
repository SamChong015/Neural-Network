#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <vector>
#include <cmath>

#include "Matrix.h"
#include "ActivationFunctions.h"
#include "LossFunctions.h"

float activationFunction(float x)
{
    return x;
}

float lossFunction(Matrix<float> predictions, Matrix<float> targets)
{
    float x = 0.0f;
    return x;
}

Matrix<float> dxLossFunction(Matrix<float> predictions, Matrix<float> targets, float batch_size)
{
    Matrix<float> temp;
    return temp;
}

class Network
{
private:

    std::vector<Matrix<float>> values;
    std::vector<Matrix<float>> weights;
    std::vector<Matrix<float>> biases;
    std::vector<int> topology;

    bool printtoConsol = true;

public:
    Network(std::vector<int> top)
        : values({}), weights({}), biases({}),  topology(top)
    {
        if (printtoConsol == true) { std::cout << "Initilizing Network Values" << std::endl; }

        for (size_t i = 0; i < top.size(); i++)
        {
            values.push_back(Matrix<float>(1, top.at(i)));
        }

        if (printtoConsol == true) { std::cout << "Initilizing Network Biases" << std::endl; }

        for (size_t i = 1; i < top.size(); i++)
        {
            biases.push_back(Matrix<float>(1, top.at(i)));
            biases.at(i-1).randomize();
        }

        if (printtoConsol == true)
        {
            std::cout << "Initilizing Network Weights" << std::endl;
        }

        for (size_t i = 0; i < top.size()-1; i++)
        {
            weights.push_back(Matrix<float>(top.at(i), top.at(i+1)));
            weights.at(i).randomize();
        }

        if (printtoConsol == true)
        {
            std::cout << "Finished Initilizing Network" << std::endl;
        }
    }

    auto FeedForward(Matrix<float>inPut)
    {
        for (size_t i = 0; i < topology.size()-1; i++)
        {
            values.at(i + 1) = values.at(i) * weights.at(i) + biases.at(i); //Add activation Function
            values.at(i + 1).applyFunction(activationFunction);
        }
        return values.back();
    }
   
    void backwardPass(std::vector<Matrix<float>> input, std::vector<Matrix<float>> target, float lossValue, int batch) 
    {
        for (int i = 0; i < input.size(); i++)
        {
            Matrix<float> outputPrev;
            outputPrev = FeedForward(input.at(i));

            Matrix<float> neuronError;
            std::vector<Matrix<float>> neuronErrors;

            for (size_t j = topology.size(); j < topology.size()-1; j--)
            {
               // neuronError = (outputPrev - target.at(i)) * dxLossFunction(input.at(i), target.at(i), batch);
            }

        }
    }

    void printNet()
    {
        std::cout << "Values" << std::endl;
        for (size_t i = 0; i < values.size(); i++)
        {
            values.at(i).printShape();
            values.at(i).printMatrix();
        }

        std::cout << "Biases" << std::endl;
        for (size_t i = 0; i < biases.size(); i++)
        {
            biases.at(i).printShape();
            biases.at(i).printMatrix();
        }

        std::cout << "Weights" << std::endl;
        for (size_t i = 0; i < weights.size(); i++)
        {
            weights.at(i).printShape();
            weights.at(i).printMatrix();
        }
    }
};

#endif

