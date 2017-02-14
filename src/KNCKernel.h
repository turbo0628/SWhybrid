#ifndef _KNCKERNEL_H
#define _KNCKERNEL_H

#include <offload.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#define __ONMIC__ __attribute__((target(mic)))

__ONMIC__ void Compute(char* query, int qlen, char* deviceDBSeq,
		int* deviceMap,
	 	size_t batchNum, int* devResult, int* deviceMatrix, int& globalCounter, int pass, int bodyQueryLen, int lastQueryLen, int* g_profile, int** g_qtable);

__ONMIC__ void pfalloc(char* query, int qlen, int threadNum, int* g_profile, int** g_qtable);
__ONMIC__ void pffree();
#endif
