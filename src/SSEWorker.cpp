/*
 * SSEWorker.cpp
 *
 *  Created on: 2014-7-18
 *      Author: lan
 */

#include "SSEWorker.h"
#include "SSEKernel.h"

#ifdef HAVE_SSSE3
#include <tmmintrin.h>
#else
#include <emmintrin.h>
#endif

#include <cstdio>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <sys/timeb.h>
#include <vector>
#include <unistd.h>
#include <omp.h>
#include <unistd.h>

using namespace std;

#define CDEPTH 4

#define BUFFERSIZE 2048
#define MAXNAMESIZE 2048

static int overflow_cnt = 0;
static int SCORE_LIMIT_7 = 0;
static int gape = -1;
static int gapoe = -1;

static int q, r;
static long qlen;
static char* sseQuery;
static BYTE* score_matrix;


SSEWorker::SSEWorker(DataPool* loader, Param* params, RecalcWorker* recalcWorker):
	Worker(loader, params, recalcWorker, 1, 16,sizeof(BYTE)),
				sseThreads(params->cpuNum),
				mapSize(4 << 20)
{
	gape = params->gapExtend;
	gapoe = params->gapOE;	
	bufferSize = 1 << 27;
	cpuFlag = true;

	score_matrix = new BYTE[32 * 32];
	score_matrix_16 = new short[32 * 32];

	SCORE_LIMIT_7 = score_limit_7;
}

SSEWorker::~SSEWorker()
{
}

void SSEWorker::setQuery(const char* in_query, const size_t queryLen)
{
	cpuFlag = true;
	this->queryLen = queryLen;
	qlen = queryLen;
	sseQuery = new char[queryLen];
	this->query = new char[queryLen];
	memcpy(sseQuery, in_query, sizeof(char) * qlen);
	memcpy(this->query, in_query, sizeof(char) * qlen);
	for( int i = 0; i < 1024; ++i){
		int sc = params->plain_matrix[i];
		score_matrix[i] = sc;
	}
}

void SSEWorker::alloc(){
	sseThreads = params->cpuNum - params->cpuReservedCores;

	initLocks();

	//of base class
	buf[0] = (BYTE*) _mm_malloc(sizeof(BYTE) * bufferSize, 64);
	buf[1] = (BYTE*) _mm_malloc(sizeof(BYTE) * bufferSize, 64);

	fillMap[0] = (int *) _mm_malloc(sizeof(int) * mapSize * 2, 64);
	fillMap[1] = (int *) _mm_malloc(sizeof(int) * mapSize * 2, 64);

	score[0] = (int *) _mm_malloc(sizeof(int) * mapSize * 16, 16);
	score[1] = (int *) _mm_malloc(sizeof(int) * mapSize * 16, 16);

	entry[0].setEntry(buf[0], fillMap[0], score[0]);
	entry[1].setEntry(buf[1], fillMap[1], score[1]);

	compEntry = entry + 1;
	auxEntry = entry;

	overflow_indices = (int *) _mm_malloc(sizeof(int) * mapSize * 16, 16);

}

void SSEWorker::free(){
	destroyLocks();
	_mm_free(score[0]);
	_mm_free(score[1]);
	_mm_free(overflow_indices);
	_mm_free(buf[0]);
	_mm_free(buf[1]);
	_mm_free(fillMap[0]);
	_mm_free(fillMap[1]);

	delete [] score_matrix;
	delete [] score_matrix_16;

}

void SSEWorker::search(){
	int globalCounter = -1;
#pragma omp parallel num_threads(sseThreads)
	Compute(
		(BYTE*) sseQuery,
		qlen,
		gapoe,
		gape,
		score_matrix,
		compEntry->getBuf(),
		compEntry->getMap(),
		compEntry->getBatchNum(), 
		compEntry->getScores(),
		overflow_indices,
		overflow_cnt,
		globalCounter,
		SCORE_LIMIT_7
	       );
	handleOverflow(
		compEntry->getBuf(),
		compEntry->getMap(),
		compEntry->getBatchNum(), 
		compEntry->getBaseIdx()
	      );
}

void SSEWorker::handleOverflow(
		char* dbSeq,
		int* map,
		int batchNum,
		size_t baseIndex
		)
{
	BYTE* dst = recalcBuf + recalcSize;
	for(int i = 0; i != overflow_cnt; ++i)
	{
		int idx = overflow_indices[i];
		{
			int batchIdx = idx / 16;
			int channel  = idx % 16;
			int pos = map[batchIdx * 2];
			int len = map[batchIdx * 2 + 1];
			char buf[len];

			char* seq = dbSeq + pos + channel;
			for(int j = 0; j != len; ++j){
				buf[j] = seq[j * 16];
				if(buf[j] > 23){
					buf[j] = 23;
				}
			}
			recalcWorker->pushOverflowSeq(buf, len, idx + baseIndex);
			
		}
	}//for
	overflow_cnt = 0;
}

void SSEWorker::showPerformance(){
	/*show SSE performance*/
	double startTimeSec = computeStartTime.time + computeStartTime.millitm / 1000.0;
	double endTimeSec = computeStopTime.time + computeStopTime.millitm / 1000.0;
	double computeTime = endTimeSec - startTimeSec;
	printf("SSE computing time %lfs, AARes %ld, GCUPs %lf\n", computeTime,
			TotalAminoAcidResidue, TotalAminoAcidResidue * queryLen
			/ computeTime / 1000000000);
}
