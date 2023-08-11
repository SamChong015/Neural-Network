#ifndef LOSSFUNCTIONS_H
#define LOSSFUNCTIONS_H

#include <iostream>
#include <vector>
#include <cmath>

#include "Matrix.h"

// Mean Squared Error (MSE) loss function
float meanSquaredError(Matrix<float> predictions, Matrix<float> targets) 
{
    float sumSquaredError = 0.0;
    for (size_t c = 0; c < predictions.getCols(); c++)
    {
        for (size_t r = 0; r < predictions.getRows(); r++)
        {
            float error = predictions.at(r, c) - targets.at(r, c);
            sumSquaredError += error * error;
        }
    }

    return sumSquaredError / predictions.getCols();
}

Matrix<float> dxMeanSquaredError(Matrix<float> predictions, Matrix<float> targets, float batch_size)
{
    return  (predictions = targets) * 2.0f / batch_size;
}

// Binary Cross-Entropy loss function
float binaryCrossEntropy(Matrix<float> predictions, Matrix<float> targets)
{
    float loss = 0.0;
    for (size_t c = 0; c < predictions.getCols(); c++)
    {
        for (size_t r = 0; r < predictions.getRows(); r++)
        {
            float prediction = predictions.at(r, c);
            float target = targets.at(r, c);

            // Avoid log(0) by adding a small epsilon value
            float epsilon = 1e-8;
            float clippedPrediction = std::max(epsilon, std::min(1.0f - epsilon, prediction));

            loss += -target * log(clippedPrediction) - (1.0 - target) * log(1.0 - clippedPrediction);
        }
    }
    return loss / predictions.getCols();
}

Matrix<float> dxBinaryCrossEntropy(Matrix<float> predictions, Matrix<float> targets, float batch_size) {
    size_t rows = predictions.getRows();
    size_t cols = predictions.getCols();
    Matrix<float> dLoss_dPredictions(rows, cols);

    for (size_t c = 0; c < cols; c++) {
        for (size_t r = 0; r < rows; r++) {
            float prediction = predictions.at(r, c);
            float target = targets.at(r, c);

            // Avoid log(0) by adding a small epsilon value
            float epsilon = 1e-8;
            float clippedPrediction = std::max(epsilon, std::min(1.0f - epsilon, prediction));

            // Calculate the derivative of the loss function with respect to clippedPrediction
            float dLoss_dClippedPrediction = (clippedPrediction - target) / (clippedPrediction * (1.0 - clippedPrediction));

            // Calculate the derivative of clippedPrediction with respect to the raw prediction (1.0)
            float dClippedPrediction_dPrediction = 1.0;

            // Calculate the overall derivative of the loss with respect to the raw prediction
            float dLoss_dPrediction = dLoss_dClippedPrediction * dClippedPrediction_dPrediction;

            // Update the dLoss_dPredictions matrix with the computed derivative
            dLoss_dPredictions.setAt(r, c, dLoss_dPrediction);
        }
    }

    // Scale the derivative by batch_size (average over the batch)
    dLoss_dPredictions = dLoss_dPredictions / batch_size;

    return dLoss_dPredictions;
}

// Sparse Categorical Cross-Entropy loss function
float sparseCategoricalCrossEntropy(Matrix<float> predictions, Matrix<float> targets) 
{
    float loss = 0.0;
    for (size_t c = 0; c < predictions.getCols(); c++)
    {
        for (size_t r = 0; r < predictions.getRows(); r++)
        {
            float target = targets.at(r, c);
            float prediction = predictions.at(target, c);

            // Avoid log(0) by adding a small epsilon value
            float epsilon = 1e-8;
            float clippedPrediction = std::max(epsilon, std::min(1.0f - epsilon, prediction));

            loss += -log(clippedPrediction);
        }
    }

    return loss / predictions.getCols();
}

Matrix<float> dxSparseCategoricalCrossEntropy(Matrix<float> predictions, Matrix<float> targets, float batch_size)
{
    Matrix<float> gradients(predictions.getRows(), predictions.getCols());

    for (size_t c = 0; c < predictions.getCols(); c++)
    {
        for (size_t r = 0; r < predictions.getRows(); r++)
        {
            float target = targets.at(r, c);
            float prediction = predictions.at(r, c);

            // Avoid log(0) by adding a small epsilon value
            float epsilon = 1e-8;
            float clippedPrediction = std::max(epsilon, std::min(1.0f - epsilon, prediction));

            // Calculate the gradient for the current element
            gradients.setAt(r, c, -1.0f / clippedPrediction * (r == static_cast<size_t>(target) ? 1.0f : 0.0f));
        }
    }
    return gradients;
}

float hingeLoss(Matrix<float> predictions, Matrix<float> targets)
{
    float loss = 0.0;

    for (size_t c = 0; c < predictions.getCols(); c++)
    {
        for (size_t r = 0; r < predictions.getRows(); r++)
        {
            float prediction = predictions.at(r,c);
            float target = targets.at(r, c);

            float margin = 1.0f - target * prediction;
            loss += std::max(0.0f, margin);
        }
    }

    return loss / predictions.getCols();
}

Matrix<float> dxHingeLoss(Matrix<float> predictions, Matrix<float> targets, float batch_size)
{
    size_t rows = predictions.getRows();
    size_t cols = predictions.getCols();
    Matrix<float> subgradient(rows, cols);

    for (size_t c = 0; c < cols; c++)
    {
        for (size_t r = 0; r < rows; r++)
        {
            float prediction = predictions.at(r, c);
            float target = targets.at(r, c);

            if (1 - target * prediction > 0)
            {
                subgradient.setAt(r, c, -target);
            }
        }
    }

    return subgradient;
}

#endif

