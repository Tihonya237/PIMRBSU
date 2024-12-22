#ifndef PARALSOLVER_H
#define PARALSOLVER_H

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

class Solver
{
private:
    //int n, maxIter;
    double eps;

    vector<int> ig, jg;
    vector<double> di, gg, ggl, ggu, b;

    vector<double> ggl_n, di_n, ggu_n, s_n;
    vector<double> q, r, p, z, temp1, temp2, r_b;

    int numThreads = 1;

    void DI();
    void multParallel(vector<double>& MV, vector<double>& vec, vector<double>& cggl, vector<double>& cggu, vector<double>& cdi);
    void mult(vector<double>& res, const vector<double>& vec, const vector<double>& cggl, const vector<double>& cggu, const vector<double>& cdi);
    double norm(const vector<double>& vec) const;
    void mult_direct(const vector<double>& d, vector<double>& y, const vector<double>& bb);
    double skal(const vector<double>& vec1, const vector<double>& vec2) const;

public:
    int n, maxIter;
    void input(string path);
    void solve(vector<double>& q, bool useParallel = false, int numThreads = 1);
    void outputSolution(std::string soulutionFile);
};
#endif

