#include <iostream>
#include <vector>

#include "Matrix.h"
#include "Network.h"

int main()
{

	std::vector<int> topology = { 2,3,3,2 };


	Matrix<float> inPut(1, 2);
	inPut.randomize();

	Matrix<float> outPut(1, 2);

	Network test(topology);

	//test.printNet();

	std::cout << "Feeding Forward" << std::endl;
	outPut = test.FeedForward(inPut);

	std::cout << "Printing Output" << std::endl;
	outPut.printMatrix();

	return 1;
}