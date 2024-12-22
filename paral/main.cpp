#include "ParalSolver.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>


using namespace std;

int main()
{
	Solver ps;

	string path = "SLAE/";
	ps.input(path);
	std::ofstream time("time_58081.csv");
    try
    {
        for (size_t i = 1; i <= 8; i++)
        {
            vector<double> q(ps.n);

            auto start = std::chrono::high_resolution_clock::now();
            ps.solve(q, true, i);
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double> duration = end - start;
            time << i << "," << duration.count() << '\n';

            ps.outputSolution("solution_58081.txt");
        }
        time.close();
    }
    catch (const std::invalid_argument& e) {
        std::cerr << e.what() << std::endl;
    }
    system("pause");

    return 0;
   
}