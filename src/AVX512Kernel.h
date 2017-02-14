#ifndef _MICKERNEL_H
#define _MICKERNEL_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>


void avx512Compute(char* query, 
		int qlen, 
		int gapoe, 
		int gape, 
		char* deviceDBSeq,
		int* deviceMap,	
		size_t batchNum, 
		int* devResult, 
		int* deviceMatrix, 
		int& globalCounter,  
		int* g_profile, 
		int** g_qtable
	    );

void pfalloc(char* query, int qlen, int threadNum, int* g_profile, int** g_qtable);
void pffree();
#endif
