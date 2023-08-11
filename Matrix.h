// Matrix Library with limited functions for use in neural networks
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cmath>
#include <cassert>
#include <iostream>
#include <functional>
#include <tuple>
#include <fstream>
#include <ctime>

float getRandomNumber() {
    // Seed the random number generator with the current time
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    // Generate a random number between 0 and RAND_MAX
    int randomInt = std::rand();

    // Scale the random number to a value between 0 and 1
    float randomDouble = static_cast<double>(randomInt) / RAND_MAX;

    return randomDouble;
}

template<typename Type>
class Matrix {

private:
    size_t cols;
    size_t rows;
    std::vector<Type> data;
    std::tuple<size_t, size_t> shape;
    bool printtoConsol = true;

public:

    /* constructors */
    Matrix(size_t rows, size_t cols)
        : cols(cols), rows(rows), data({}) {

        data.resize(cols * rows, Type());  // init empty vector for data
        shape = { rows, cols };

    }
    Matrix() : cols(0), rows(0), data({}) { shape = { rows, cols }; };

    void printShape() {
        std::cout << "Matrix Size([" << rows << ", " << cols << "])" << std::endl;
    }

    size_t getRows()
    {
        return rows;
    }

    size_t getCols()
    {
        return cols;
    }

    std::tuple<> getShape()
    {
        return shape;
    }

    void printMatrix() {
        for (size_t r = 0; r < rows; r++)
        {
            for (size_t c = 0; c < cols; c++)
            {
                std::cout << " " << at(r, c);
            }
            std::cout << '\n';
        }
        std::cout << std::endl;
    }

    Type at(size_t row, size_t col) {
        return data[row * cols + col];
    }

    void setAt(size_t row, size_t col, Type val) {
        data[row * cols + col] = val;
    }

    Matrix operator+(Matrix& other) {

        //if (printToConsol == true) { std::cout << "Adding two Matrices" << std::endl; }
        assert(cols == other.cols && rows == other.rows);

        auto result = Matrix<Type>(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result.setAt(r, c, (at(r, c) + other.at(r, c)));
            }
        }
        return result;
    }

    Matrix operator-(Matrix& other) {

        //if (printtoConsol == true) { std::cout << "Subtracting two Matrices" << std::endl; }
        assert(cols == other.cols && rows == other.rows);

        auto result = Matrix<Type>(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result.setAt(r, c, (at(r, c) - other.at(r, c)));
            }
        }
        return result;
    }

    Matrix operator*(Matrix& other) {
        int resultRows = rows;
        int resultColumns = other.cols;

        //if (printToConsol == true) { std::cout << "Multiplying two Matrices" << std::endl; }
        assert(cols == other.rows);

        auto result = Matrix<Type>(resultRows, resultColumns);

        for (int i = 0; i < resultRows; ++i) {
            for (int j = 0; j < resultColumns; ++j) {
                int sum = 0;
                for (int k = 0; k < cols; ++k) {
                    sum += at(i, k) * other.at(k, j);
                }
                result.setAt(i, j, sum);
            }
        }

        return result;
    }

    Matrix operator*(const Type& other) {
        //if (printToConsol == true) { std::cout << "Multiplying a Matrix by a Scaler" << std::endl; }

        auto result = Matrix<Type>(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result.setAt(r, c, (at(r, c) * other));
            }
        }
        return result;
    }

    Matrix operator/(const Type& other) {
        //if (printToConsol == true) { std::cout << "Dividing a Matrix by a Scaler" << std::endl; }

        auto result = Matrix<Type>(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result.setAt(r, c, (at(r, c) / other));
            }
        }
        return result;
    }

    Matrix elementWiseMultiplication(Matrix& other) {
        //if (printToConsol == true) { std::cout << "Multiplying two Matrices by Elements" << std::endl; }
        assert(cols == other.cols && rows == other.rows);

        auto result = Matrix<Type>(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                result.setAt(r, c, (at(r, c) * other.at(r, c)));
            }
        }
        return result;
    }

    Matrix transformation() {
        int resultRows = cols;
        int resultColumns = rows;
        auto result = Matrix<Type>(resultRows, resultColumns);

        for (int r = 0; r < resultRows; ++r) {
            for (int c = 0; c < resultColumns; ++c) {
                result.setAt(r, c, (at(c, r)));
            }
        }
        return result;
    }

    void applyFunction(std::function<Type(Type)> func)
    {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                setAt(r, c, func(at(r, c)));
            }
        }
    }

    void randomize()
    {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                setAt(r, c, getRandomNumber());
            }
        }
    }

    bool writeToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Failed to open the file: " << filename << std::endl;
            return false;
        }

        int rows_, cols_;

        file << rows_ << " " << cols_ << std::endl;
        for (int row = 0; row < rows_; ++row) {
            for (int col = 0; col < cols_; ++col) {
                file << (*this)(row, col) << " ";
            }
            file << std::endl;
        }

        file.close();
        return true;
    }

    bool readFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file) {
            std::cerr << "Failed to open the file: " << filename << std::endl;
            return false;
        }

        int rows_, cols_;
        file >> rows_ >> cols_;
        if (rows_ <= 0 || cols_ <= 0) {
            std::cerr << "Invalid matrix dimensions in the file: " << filename << std::endl;
            file.close();
            return false;
        }

        Matrix temp(rows_, cols_);
        for (int row = 0; row < rows_; ++row) {
            for (int col = 0; col < cols_; ++col) {
                file >> temp(row, col);
            }
        }

        rows = rows_;
        cols = cols_;
        data = std::move(temp.data);

        file.close();
        return true;
    }
};

#endif