#include <matrix.h>
#include "mex.h"
#include <cmath>
#include <omp.h>
#define cmin(a,b) ((a) < (b) ? (a) : (b))
#define cmax(a,b) ((a) > (b) ? (a) : (b))

#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>
#include <unordered_map>
using namespace std;


inline int index(int i, int j, int numRows) // 2D 0-indexed to C
{
	return j*numRows + i;
}

inline double euclidean(double x1, double y1, double x2, double y2)
{
	return sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
}

double boxiou(double l1, double t1, double w1, double h1, double l2, double t2, double w2, double h2)
{
	double area1 = w1 * h1;
	double area2 = w2 * h2;

	double x_overlap = cmax(0, cmin(l1 + w1, l2 + w2) - cmax(l1, l2));
	double y_overlap = cmax(0, cmin(t1 + h1, t2 + h2) - cmax(t1, t2));
	double intersectionArea = x_overlap*y_overlap;
	double unionArea = area1 + area2 - intersectionArea;
	double iou = intersectionArea / unionArea;
	return iou;
}

// cmin cost bipartite matching via shortest augmenting paths
//
// Code from https://github.com/jaehyunp/
//
// This is an O(n^3) implementation of a shortest augmenting path
// algorithm for finding cmin cost perfect matchings in dense
// graphs.  In practice, it solves 1000x1000 problems in around 1
// second.
//
//   cost[i][j] = cost for pairing left node i with right node j
//   Lmate[i] = index of right node that left node i pairs with
//   Rmate[j] = index of left node that right node j pairs with
//
// The values in cost[i][j] may be positive or negative.  To perform
// cmaximization, simply negate the cost[][] matrix.



typedef vector<double> VD;
typedef vector<VD> VVD;
typedef vector<int> VI;

double cminCostMatching(const VVD &cost, VI &Lmate, VI &Rmate) {
	int n = int(cost.size());

	// construct dual feasible solution
	VD u(n);
	VD v(n);
	for (int i = 0; i < n; i++) {
		u[i] = cost[i][0];
		for (int j = 1; j < n; j++) u[i] = cmin(u[i], cost[i][j]);
	}
	for (int j = 0; j < n; j++) {
		v[j] = cost[0][j] - u[0];
		for (int i = 1; i < n; i++) v[j] = cmin(v[j], cost[i][j] - u[i]);
	}

	// construct primal solution satisfying complementary slackness
	Lmate = VI(n, -1);
	Rmate = VI(n, -1);
	int mated = 0;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (Rmate[j] != -1) continue;
			if (fabs(cost[i][j] - u[i] - v[j]) < 1e-10) {
				Lmate[i] = j;
				Rmate[j] = i;
				mated++;
				break;
			}
		}
	}

	VD dist(n);
	VI dad(n);
	VI seen(n);

	// repeat until primal solution is feasible
	while (mated < n) {

		// find an unmatched left node
		int s = 0;
		while (Lmate[s] != -1) s++;

		// initialize Dijkstra
		fill(dad.begin(), dad.end(), -1);
		fill(seen.begin(), seen.end(), 0);
		for (int k = 0; k < n; k++)
			dist[k] = cost[s][k] - u[s] - v[k];

		int j = 0;
		while (true) {

			// find closest
			j = -1;
			for (int k = 0; k < n; k++) {
				if (seen[k]) continue;
				if (j == -1 || dist[k] < dist[j]) j = k;
			}
			seen[j] = 1;

			// tercmination condition
			if (Rmate[j] == -1) break;

			// relax neighbors
			const int i = Rmate[j];
			for (int k = 0; k < n; k++) {
				if (seen[k]) continue;
				const double new_dist = dist[j] + cost[i][k] - u[i] - v[k];
				if (dist[k] > new_dist) {
					dist[k] = new_dist;
					dad[k] = j;
				}
			}
		}

		// update dual variables
		for (int k = 0; k < n; k++) {
			if (k == j || !seen[k]) continue;
			const int i = Rmate[k];
			v[k] += dist[k] - dist[j];
			u[i] -= dist[k] - dist[j];
		}
		u[s] += dist[j];

		// augment along path
		while (dad[j] >= 0) {
			const int d = dad[j];
			Rmate[j] = Rmate[d];
			Lmate[Rmate[j]] = j;
			j = d;
		}
		Rmate[j] = s;
		Lmate[s] = j;

		mated++;
	}

	double value = 0;
	for (int i = 0; i < n; i++)
		value += cost[i][Lmate[i]];

	return value;
}

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{

	double* gtXi = mxGetPr(mxGetField(prhs[0], 0, "Xi"));
	double* gtYi = mxGetPr(mxGetField(prhs[0], 0, "Yi"));
	double* gtW = mxGetPr(mxGetField(prhs[0], 0, "W"));
	double* gtH = mxGetPr(mxGetField(prhs[0], 0, "H"));

	int field_num_gtXgp = mxGetFieldNumber(prhs[0], "Xgp");
	int field_num_gtYgp = mxGetFieldNumber(prhs[0], "Ygp");
	double *gtXgp, *gtYgp;
	if (field_num_gtXgp != -1) gtXgp = mxGetPr(mxGetField(prhs[0], 0, "Xgp"));
	if (field_num_gtYgp != -1) gtYgp = mxGetPr(mxGetField(prhs[0], 0, "Ygp"));

	double* stXi = mxGetPr(mxGetField(prhs[1], 0, "Xi"));
	double* stYi = mxGetPr(mxGetField(prhs[1], 0, "Yi"));
	double* stW = mxGetPr(mxGetField(prhs[1], 0, "W"));
	double* stH = mxGetPr(mxGetField(prhs[1], 0, "H"));

	int field_num_stXgp = mxGetFieldNumber(prhs[1], "Xgp");
	int field_num_stYgp = mxGetFieldNumber(prhs[1], "Ygp");
	double *stXgp, *stYgp;
	if (field_num_stXgp != -1) stXgp = mxGetPr(mxGetField(prhs[1], 0, "Xgp"));
	if (field_num_stYgp != -1) stYgp = mxGetPr(mxGetField(prhs[1], 0, "Ygp"));

	double threshold = (double)mxGetScalar(prhs[2]);
	bool world = (bool)mxGetScalar(prhs[3]);

	const int  *dimGT, *dimST;
	dimGT = (int*)mxGetDimensions(mxGetField(prhs[0], 0, "Xi"));
	dimST = (int*)mxGetDimensions(mxGetField(prhs[1], 0, "Xi"));
	int Fgt = dimGT[0], Ngt = dimGT[1], F = dimST[0], N = dimST[1];

	plhs[0] = mxCreateDoubleMatrix(1, F, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(1, F, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(1, F, mxREAL);
	plhs[3] = mxCreateDoubleMatrix(1, F, mxREAL);
	plhs[4] = mxCreateDoubleMatrix(1, F, mxREAL);
	plhs[5] = mxCreateDoubleMatrix(F, Ngt, mxREAL);
	plhs[6] = mxCreateDoubleMatrix(F, Ngt, mxREAL);
	plhs[7] = mxCreateDoubleMatrix(F, Ngt, mxREAL);
	plhs[8] = mxCreateDoubleMatrix(F, N, mxREAL);
	plhs[9] = mxCreateDoubleMatrix(F, Ngt, mxREAL);

	double* mmeOut = mxGetPr(plhs[0]);
	double* cOut = mxGetPr(plhs[1]);
	double* fpOut = mxGetPr(plhs[2]);
	double* mOut = mxGetPr(plhs[3]);
	double* gOut = mxGetPr(plhs[4]);
	double* dOut = mxGetPr(plhs[5]);
	double* iousOut = mxGetPr(plhs[6]);
	double* alltrackedOut = mxGetPr(plhs[7]);
	double* allfalseposOut = mxGetPr(plhs[8]);
	double* MOut = mxGetPr(plhs[9]);

	double INF = 1e9;
	
	
	vector<unordered_map<int,int>> gtInd(Fgt);
	vector<unordered_map<int,int>> stInd(F);
	vector<unordered_map<int,int>> M(F);
	vector<int> mme(F, 0); // ID Switchtes(mismatches)
	vector<int> c(F, 0); // matches found
	vector<int> fp(F, 0); // false positives
	vector<int> m(F, 0); // misses = false negatives
	vector<int> g(F, 0); // gt count for each frame
	vector<vector<int>> d(F, vector<int>(Ngt, 0)); // all distances
	vector<vector<double>> ious(F, vector<double>(Ngt, INF)); // all overlaps
	vector<vector<int>> alltracked(F, vector<int>(Ngt, 0));
	vector<vector<int>> allfalsepos(F, vector<int>(N, 0));


	for (int i = 0; i < Fgt; i++) {
		for (int j = 0; j < Ngt; j++) {
			if (!(!gtXi[index(i, j, Fgt)]))
				gtInd[i][j] = true;
		}
	}

	for (int i = 0; i < F; i++) {
		for (int j = 0; j < N; j++) {
			if (!(!stXi[index(i, j, F)]))
				stInd[i][j] = true;
		}
		for (unordered_map<int, int>::iterator it = gtInd[i].begin(); it != gtInd[i].end(); it++) g[i]++;
	}

	for (int t = 0; t < F; t++)
	{
		if ((t + 1) % 1000 == 0) mexEvalString("fprintf('.');");  // print every 1000th frame

		if (t > 0)
		{
			vector<int> mappings;
			for (unordered_map<int,int>::iterator it = M[t - 1].begin(); it != M[t - 1].end(); it++) mappings.push_back(it->first);

			for (int k = 0; k < mappings.size(); k++)
			{
				unordered_map<int, int>::const_iterator foundGtind = gtInd[t].find(mappings[k]);
				unordered_map<int, int>::const_iterator foundStind = stInd[t].find(M[t - 1][mappings[k]]);

				if (foundGtind != gtInd[t].end() && foundStind != stInd[t].end())
				{
					bool matched = false;
					if (world)
					{
						double gtx, gty, stx, sty;
						gtx = gtXi[index(t, mappings[k], Fgt)];
						gty = gtYi[index(t, mappings[k], Fgt)];
						stx = stXi[index(t, M[t - 1][mappings[k]], F)];
						sty = stYi[index(t, M[t - 1][mappings[k]], F)];
						double dist = euclidean(gtx, gty, stx, sty);
						matched = (dist <= threshold);
					}
					else
					{
						double gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight;
						int indgt = index(t, mappings[k], Fgt), indst = index(t, M[t - 1][mappings[k]], F);
						gtleft = gtXi[indgt] - gtW[indgt] / 2;
						gttop = gtYi[indgt] - gtH[indgt];
						gtwidth = gtW[indgt];
						gtheight = gtH[indgt];
						stleft = stXi[indst] - stW[indst] / 2;
						sttop = stYi[indst] - stH[indst];
						stwidth = stW[indst];
						stheight = stH[indst];
						double iou = boxiou(gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight);
						matched = (iou >= threshold);
					}

					if (matched) M[t][mappings[k]] = M[t - 1][mappings[k]];
				}
			}
		}

		vector<int> unmappedGt, unmappedEs, stindt, findm;
		for (unordered_map<int,int>::iterator it = gtInd[t].begin(); it != gtInd[t].end(); it++) {
			unordered_map<int, int>::const_iterator found = M[t].find(it->first);
			if (found==M[t].end()) unmappedGt.push_back(it->first);
		}
		for (unordered_map<int,int>::iterator it = M[t].begin(); it != M[t].end(); it++) findm.push_back(it->second);
		for (unordered_map<int,int>::iterator it = stInd[t].begin(); it != stInd[t].end(); it++) stindt.push_back(it->first);

		sort(stindt.begin(), stindt.end());
		sort(findm.begin(), findm.end());
		set_difference(stindt.begin(), stindt.end(), findm.begin(), findm.end(), inserter(unmappedEs, unmappedEs.end()));

		int squareSize = cmax(unmappedGt.size(), unmappedEs.size());
		vector<vector<double>> alldist(squareSize, vector<double>(squareSize, INF));

		for (int i = 0; i < unmappedGt.size(); i++)
		{
			for (int j = 0; j < unmappedEs.size(); j++)
			{
				int o = unmappedGt[i];
				int e = unmappedEs[j];
				if (world)
				{
					double gtx, gty, stx, sty;
					gtx = gtXi[index(t, o, Fgt)];
					gty = gtYi[index(t, o, Fgt)];
					stx = stXi[index(t, e, F)];
					sty = stYi[index(t, e, F)];
					double dist = euclidean(gtx, gty, stx, sty);
					if (dist <= threshold) alldist[o][e] = dist;
				}
				else
				{
					double gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight;
					int indgt = index(t, o, Fgt), indst = index(t, e, F);
					gtleft = gtXi[indgt] - gtW[indgt] / 2;
					gttop = gtYi[indgt] - gtH[indgt];
					gtwidth = gtW[indgt];
					gtheight = gtH[indgt];
					stleft = stXi[indst] - stW[indst] / 2;
					sttop = stYi[indst] - stH[indst];
					stwidth = stW[indst];
					stheight = stH[indst];
					double iou = boxiou(gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight);
					if (iou >= threshold)
						alldist[i][j] = 1 - iou;
				}
			}
		}

		vector<int> Lmate, Rmate;
		cminCostMatching(alldist, Lmate, Rmate);

		for (int k = 0; k < Lmate.size(); k++) {
			if (alldist[k][Lmate[k]] == INF) continue;
			M[t][unmappedGt[k]] = unmappedEs[Lmate[k]];
		}

		vector<int> curtracked, alltrackers, mappedtrackers, falsepositives, set1;

		for (unordered_map<int,int>::iterator it = M[t].begin(); it != M[t].end(); it++) {
			curtracked.push_back(it->first);
			set1.push_back(it->second);
		}
		for (unordered_map<int,int>::iterator it = stInd[t].begin(); it != stInd[t].end(); it++) alltrackers.push_back(it->first);

		sort(set1.begin(), set1.end());
		sort(alltrackers.begin(), alltrackers.end());
		set_intersection(set1.begin(), set1.end(), alltrackers.begin(), alltrackers.end(), inserter(mappedtrackers, mappedtrackers.begin()));
		set_difference(alltrackers.begin(), alltrackers.end(), mappedtrackers.begin(), mappedtrackers.end(), inserter(falsepositives, falsepositives.end()));

		for (unordered_map<int,int>::iterator it = M[t].begin(); it != M[t].end(); it++) alltracked[t][it->first] = it->second;
		for (int k = 0; k < falsepositives.size(); k++) allfalsepos[t][falsepositives[k]] = falsepositives[k];

		//  mismatch errors
		if (t > 0)
		{
			for (int k = 0; k < curtracked.size(); k++)
			{
				int ct = curtracked[k];
				int lastnonempty = -1;
				for (int j = t - 1; j >= 0; j--) {
					if (M[j].find(ct) != M[j].end()) {
						lastnonempty = j; break;
					}
				}
				if (gtInd[t-1].find(ct)!=gtInd[t-1].end() && lastnonempty != -1)
				{
					int mtct = -1, mlastnonemptyct = -1;
					if (M[t].find(ct) != M[t].end()) mtct = M[t][ct];
					if (M[lastnonempty].find(ct) != M[lastnonempty].end()) mlastnonemptyct = M[lastnonempty][ct];

					if (mtct != mlastnonemptyct)
						mme[t]++;
				}
			}
		}

		c[t] = curtracked.size();
		for (int k = 0; k < curtracked.size(); k++)
		{
			int ct = curtracked[k];
			int eid = M[t][ct];
			if (world)
			{
				double gtx, gty, stx, sty;
				gtx = gtXi[index(t, ct, Fgt)];
				gty = gtYi[index(t, ct, Fgt)];
				stx = stXi[index(t, eid, F)];
				sty = stYi[index(t, eid, F)];
				d[t][ct] = euclidean(gtx, gty, stx, sty);
			}
			else
			{
				double gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight;
				int indgt = index(t, ct, Fgt), indst = index(t, eid, F);
				gtleft = gtXi[indgt] - gtW[indgt] / 2;
				gttop = gtYi[indgt] - gtH[indgt];
				gtwidth = gtW[indgt];
				gtheight = gtH[indgt];
				stleft = stXi[indst] - stW[indst] / 2;
				sttop = stYi[indst] - stH[indst];
				stwidth = stW[indst];
				stheight = stH[indst];

				ious[t][ct] = boxiou(gtleft, gttop, gtwidth, gtheight, stleft, sttop, stwidth, stheight);
			}
		}

		for (unordered_map<int,int>::iterator it = stInd[t].begin(); it != stInd[t].end(); it++) fp[t]++;
		fp[t] -= c[t];
		m[t] = g[t] - c[t];
	}

	for (int k = 0; k < F; k++) {
		mmeOut[k] = mme[k];
		cOut[k] = c[k];
		fpOut[k] = fp[k];
		gOut[k] = g[k];
		mOut[k] = m[k];
	}


	for (int i = 0; i < F; i++) {
		for (int j = 0; j < Ngt; j++) {
			dOut[index(i, j, F)] = d[i][j];
			iousOut[index(i, j, F)] = (ious[i][j] == INF ? mxGetInf() : ious[i][j]);
			alltrackedOut[index(i, j, F)] = alltracked[i][j];
		}

		for (unordered_map<int, int>::iterator it = M[i].begin(); it != M[i].end(); it++) {
			int j = it->first;
			MOut[index(i, j, F)] = M[i][j] + 1; // matlab indexed
		}

		for (int j = 0; j < N; j++) {
			allfalseposOut[index(i, j, F)] = allfalsepos[i][j];
		}
	}

}