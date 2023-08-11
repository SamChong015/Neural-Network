#ifndef ACTIVATIONFUNCTIONS_H
#define ACTIVATIONFUNCTIONS_H

#include <cmath>

float alpha = 0.8f; //Temp value, change as needed for LeakyRLU

float Sigmoid(float x) //For Binary Classification Problems
{
    return 1.0f / (1.0f + std::exp(-x));
}

float dxSigmoid(float x)
{
    float sigmoid_x = Sigmoid(x);
    return sigmoid_x * (1.0f - sigmoid_x);
}

float HyperbolicTan(float x) //For Hidden Layers
{
    return std::tanh(x);
}

float dxHyperbolicTan(float x)
{
    float tanh_x = std::tanh(x);
    return 1 - tanh_x * tanh_x;
}

float LeakyRLU(float x) //For Image Classification, Object Detection, Natural Language Processing
{
    if (x >= 0) {
        return x;
    }
    else {
        return alpha * x;
    }
}

float dxLeakyRLU(float x)
{
    if (x >= 0) {
        return 1.0;
    }
    else {
        return alpha;
    }
}
#endif