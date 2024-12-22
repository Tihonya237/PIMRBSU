#include "ParalSolver.h"
#include "omp.h"

void Solver::input(std::string path)
{
	ifstream input(path + "kuslau.txt");
	input >> n >> eps >> maxIter;
	input.close();

	ig.resize(n + 1);

	ifstream ig_f(path + "ig.txt");
	for (int i = 0; i < n + 1; i++)
		ig_f >> ig[i];
	ig_f.close();

	jg.resize(ig[n]);
	gg.resize(ig[n]);

	di.resize(n);
	b.resize(n);

	ifstream jg_f(path + "jg.txt");
	for (int i = 0; i < ig.size(); i++)
		jg_f >> jg[i];
	jg_f.close();

	ifstream di_f(path + "di.txt");
	for (int i = 0; i < n; i++)
		di_f >> di[i];
	di_f.close();

	ifstream gg_f(path + "gg.txt");
	for (int i = 0; i < n; i++)
		gg_f >> gg[i];
	gg_f.close();

	ifstream b_f(path + "pr.txt");
	for (int i = 0; i < n; i++)
		b_f >> b[i];
	b_f.close();

}

void Solver::multParallel(vector<double>& MV, vector<double>& vec, vector<double>& cggl, vector<double>& cggu, vector<double>& cdi)
{
	int k, rank, adr;
	vector<double> MVomp((numThreads - 1) * MV.size());

#pragma omp parallel private(i, j, k, rank, adr) num_threads(numThreads)
	{
#pragma omp for 
		for (int i = 0; i < n; i++)
			MV[i] = 0.;

#pragma omp for 
		for (int i = 0; i < n * (numThreads - 1); i++)
			MVomp[i] = 0.;

#pragma omp for 
		for (int i = 0; i < n; i++)
		{
			rank = omp_get_thread_num();

			if (rank == 0)
			{
				MV[i] = di[i] * vec[i];
				for (int j = ig[i]; j <= ig[i + 1] - 1; j++)
				{
					k = jg[j];
					MV[i] += ggl[j] * vec[k];
					MV[k] += ggl[j] * vec[i];
				}
			}
			else
			{
				adr = (rank - 1) * n;
				MVomp[adr + i] = di[i] * vec[i];
				for (int j = ig[i]; j <= ig[i + 1] - 1; j++)
				{
					k = jg[j];
					MVomp[adr + i] += ggl[j] * vec[k];
					MVomp[adr + k] += ggl[j] * vec[i];
				}
			}
		}

#pragma omp for
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < numThreads - 1; j++)
				MV[i] += MVomp[j * n + i];
		}
	}
}

void Solver::mult(vector<double>& res, const vector<double>& vec, const vector<double>& cggl, const vector<double>& cggu, const vector<double>& cdi)
{
	for (int i = 0; i < n; i++) {
		int k0 = ig[i];
		int k1 = ig[i + 1];
		res[i] = cdi[i] * vec[i];
		for (int k = k0; k < k1; k++) {
			int j = jg[k];
			res[i] += vec[j] * cggl[k];
			res[j] += vec[i] * cggu[k];
		}
	}
}

double Solver::norm(const vector<double>& vec) const
{
	double sum = 0;
	for (int i = 0; i < n; i++) {
		sum += vec[i] * vec[i];
	}
	return sqrt(sum);
}

void Solver::mult_direct(const vector<double>& d, vector<double>& y, const vector<double>& bb)
{
	int i;
#pragma omp parallel for num_threads(numThreads)
	for (i = 0; i < n; i++)
		y[i] = d[i] * bb[i];
}

double Solver::skal(const vector<double>& vec1, const vector<double>& vec2) const
{
	double sum = 0;
	int i;
#pragma omp parallel for reduction (+:sum) num_threads(numThreads) 
	for (i = 0; i < n; i++) {
		sum += vec1[i] * vec2[i];
	}
	return sum;
}

void Solver::solve(vector<double>& q, bool useParallel, int numThreads)
{
	r.resize(n);
	z.resize(n);
	p.resize(n);
	q.resize(n);
	di_n.resize(n);
	s_n.resize(n);
	temp1.resize(n);
	temp2.resize(n);
	r_b.resize(n);

	ggl.resize(ig[n]);
	ggu.resize(ig[n]);
	ggl_n.resize(ig[n]);
	ggu_n.resize(ig[n]);

	ggl = gg;
	ggu = gg;

	if (useParallel) {
		this->numThreads = numThreads;
	}

#pragma omp parallel for num_threads(numThreads) 
	for (int i = 0; i < n; i++) {
		di_n[i] = sqrt(di[i]);
		s_n[i] = 1. / di_n[i];
	}

	if (useParallel) {
		multParallel(temp1, q, ggl, ggu, di);
	}
	else {
		mult(temp1, q, ggl, ggu, di);
	}

	int i;
	for (i = 0; i < n; i++)
		temp2[i] = b[i] - temp1[i];
	double normR0 = norm(temp2);

	mult_direct(s_n, r, temp2);
	mult_direct(s_n, z, r);

	if (useParallel) {
		multParallel(temp1, z, ggl, ggu, di);
	}
	else {
		mult(temp1, z, ggl, ggu, di);
	}
	mult_direct(s_n, p, temp1);

	double nev = skal(r, r);
	int k;
	double skal1, skal2;
	for (k = 0; k < maxIter && nev > eps; k++) {
		skal1 = skal(p, r);
		skal2 = skal(p, p);

		double alfa = skal1 / skal2;
		for (i = 0; i < n; i++) {
			q[i] += alfa * z[i];
			r[i] -= alfa * p[i];
		}
		mult_direct(s_n, temp1, r);

		if (useParallel) {
			multParallel(temp2, temp1, ggl, ggu, di);
		}
		else {
			mult(temp2, temp1, ggl, ggu, di);
		}

		mult_direct(s_n, temp1, temp2);
		skal1 = skal(p, temp1);
		double beta = -skal1 / skal2;
		mult_direct(s_n, temp2, r);

#pragma omp parallel for num_threads(numThreads)
		for (i = 0; i < n; i++) {
			z[i] = temp2[i] + beta * z[i];
			p[i] = temp1[i] + beta * p[i];
		}

		mult_direct(di_n, r_b, r);
		nev = skal(r, r);
	}

	std::cout << "nev = " << skal(r, r) << std::endl;
	std::cout << "k = " << k << std::endl;

	this->q = q;

}

void Solver::outputSolution(std::string path)
{
	std::ofstream file(path);
	for (size_t i = 0; i < n; i++) {
		file << q[i] << '\n';
	}
	file.close();
}

