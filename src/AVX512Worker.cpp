#include "AVX512Worker.h"
#include "AVX512Kernel.h"

#include <cstdio>
#include <string.h>
#include <omp.h>
#include <unistd.h>
#include <stdint.h>
#include <algorithm>
#include <sys/timeb.h>

#define CDEPTH 4

static const int gape = 2;
static const int gapoe = 12;
const int restrictedQueryLen = 900;
static int pass, bodyQueryLen, lastQueryLen;

AVX512Worker::AVX512Worker(DataPool* loader, Param* params, RecalcWorker* recalcWorker):
	Worker(loader, params, recalcWorker, 1, 16, sizeof(char)),
	nThreads(params->cpuNum / 2),
	mapSize(0)
{
	bufferSize = (1 << 27) ;
	mapSize = bufferSize / 16 / 16;


}

AVX512Worker::~AVX512Worker()
{
}

void AVX512Worker::packDB(){
	Entry* pEntry = entry + !bufFlag;

	fillBatchNum = 0;
	char*	bufWorkp = dbSeqBuf;
	char*  	dstWorkp = pEntry->getBuf();
	int* 	pMap     = pEntry->getMap(); 
	batchInfo* infoData = info;
	int curPos = 0;
	int curSize = 0;

	/*copy out the buffer and transfer symbol to code*/
	this->TotalAminoAcidResidue += filledSize;

		//Vectorization 
	for (int idx = 0; idx < infoSize; ++idx) {
		int num = infoData[idx].numSeqs;
		int len = infoData[idx].seqLen;
		/*limited length for intel MIC and 32-bit version CUDA kernel*/
		if(len >= 3072){
			for(int i = 0; i != num; ++i){
				recalcWorker->pushOverflowSeq((char*) bufWorkp, len, fillBaseIndex + fillBatchNum * 16 + i);
				bufWorkp += len;
			}
			continue;
		}

		int pass = num / 16;
		for (int i = 0; i != pass; ++i) {
			for (int j = 0; j != len; j++) {
				for(int k = 0; k != 16; ++k){
					*(dstWorkp++)= bufWorkp[j + k * len];
				}
			}
			*(pMap++) = curPos;
			*(pMap++) = len;
			++fillBatchNum;
			curPos 	 += 16 * len / typeSize;
			bufWorkp += 16 * len;
		}
	}
	pEntry->setBatchNum(fillBatchNum);
	pEntry->setBaseIdx(fillBaseIndex);
	pEntry->setResultSize(fillBatchNum * 16);

}
void AVX512Worker::setQuery(const char* query, const size_t qlen)
{
	this->queryLen = qlen;
	this->query = new char[qlen];
	memcpy(this->query, query, sizeof(char) * qlen);
	if(queryLen < 3072)
		nThreads /= 2;
#if 0
	if(qlen >= restrictedQueryLen){
		pass = (qlen + threshold - 1) 
			/  threshold;
		bodyQueryLen = (qlen + pass - 1) / pass;
		lastQueryLen = qlen - bodyQueryLen * (pass - 1);
		printf("long query length %d, pass %d, major length %d, last length %d, average length %d\n", qlen,pass, bodyQueryLen, lastQueryLen, (bodyQueryLen * (pass - 1) + lastQueryLen) / pass);
	}
#endif
}

void AVX512Worker::search(){
	int globalCounter = -1;
#pragma omp parallel num_threads(nThreads)
	avx512Compute(
		this->query,
		this->queryLen,
		gapoe,
		gape,
		compEntry->getBuf(),
		compEntry->getMap(),
		compEntry->getBatchNum(), 
		compEntry->getScores(),
		plainMatrix,
		globalCounter,
		_g_profile,
		_g_qtable
	       );
}


void AVX512Worker::asyncCopy(){

}


void AVX512Worker::alloc(){
	buf[0] = (char*) _mm_malloc(sizeof(char) * bufferSize, 16);
	buf[1] = (char*) _mm_malloc(sizeof(char) * bufferSize, 16);

	fillMap[0] = (int*) _mm_malloc(sizeof(int) * mapSize * 2, 64);
	fillMap[1] = (int*) _mm_malloc(sizeof(int) * mapSize * 2, 64);

	score[0] = (int *) _mm_malloc(sizeof(int) * mapSize * 16, 64);
	score[1] = (int *) _mm_malloc(sizeof(int) * mapSize * 16, 64);

	_g_profile = (int*) _mm_malloc(sizeof(int) * 4 * 16 * 32 * nThreads, 64);
	_g_qtable  = (int**) _mm_malloc(sizeof(int*) * queryLen * nThreads, 64);


	int** qtable = _g_qtable;
	int*  profile = _g_profile;
	for(int j = 0; j != nThreads; ++j){
		for (int i = 0; i < queryLen; ++i){
			qtable[i] = profile + 64 * query[i];
		}
		profile += 4 * 16 * 32;
		qtable  += queryLen;
	}

	entry[0].setEntry(buf[0], fillMap[0], score[0]);
	entry[1].setEntry(buf[1], fillMap[1], score[1]);

	compEntry = entry + 1;
	auxEntry = entry;

	plainMatrix = (int*) _mm_malloc(sizeof(int) * 1024, 64);

	for (int i = 0; i < 32; ++i)
		for (int j = 0; j < 32; ++j)
			plainMatrix[i * 32 + j] = matrix[i][j];
}

void AVX512Worker::free(){
	_mm_free(fillMap[0]);
	_mm_free(fillMap[1]);
	_mm_free(score[0]);
	_mm_free(score[1]);
	_mm_free(buf[0]);
	_mm_free(buf[1]);
	_mm_free(_g_profile);
	_mm_free(_g_qtable);
	_mm_free(plainMatrix);
}


void AVX512Worker::showPerformance(){
	double startTimeSec = computeStartTime.time + computeStartTime.millitm / 1000.0;
	double endTimeSec = computeStopTime.time + computeStopTime.millitm / 1000.0;
	double computeTime = endTimeSec - startTimeSec;
	printf("MIC computing time %lfs, AARes %ld, GCUPs %lf\n", computeTime, 
			TotalAminoAcidResidue, TotalAminoAcidResidue * queryLen
			/ computeTime / 1000000000);
}
