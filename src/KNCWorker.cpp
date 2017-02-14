#include "KNCWorker.h"
#include "KNCKernel.h"

#include <cstdio>
#include "offload.h"
#include <string.h>
#include <omp.h>
#include <algorithm>
#include <sys/timeb.h>

#define REUSE alloc_if(0) free_if(0)

#define __ONMIC__ __attribute__((target(mic)))

#define ALLOC alloc_if(1) free_if(0)
#define FREE alloc_if(0) free_if(1)
#define CDEPTH 4

const int threshold = 750;
__ONMIC__ static const int gape = 2;
__ONMIC__ static const int gapoe = 12;
__ONMIC__ const int restrictedQueryLen = 900;
__ONMIC__ static int pass, bodyQueryLen, lastQueryLen;

KNCWorker::KNCWorker(DataPool* loader, Param* params, RecalcWorker* recalcWorker, int deviceId):
	Worker(loader, params, recalcWorker, 1, 16, sizeof(char)),
	mapSize(0),
	id(deviceId)
{
	id = deviceId;
	bufferSize = (1 << 27) ;
	mapSize = bufferSize / 16 / 16;

	/*buf[0] and buf[1] for assembled database sequence residues*/

	for (int i = 0; i < 32; ++i)
		for (int j = 0; j < 32; ++j)
			plainMatrix[i * 32 + j] = matrix[i][j];

}

KNCWorker::~KNCWorker()
{
}

void KNCWorker::setQuery(const char* query, const size_t qlen)
{
	this->queryLen = qlen;
	this->query = new char[qlen];
	memcpy(this->query, query, sizeof(char) * qlen);

#if 1
	if(qlen >= restrictedQueryLen){
		pass = (qlen + threshold - 1) 
			/  threshold;
		bodyQueryLen = (qlen + pass - 1) / pass;
		lastQueryLen = qlen - bodyQueryLen * (pass - 1);
		printf("long query length %d, pass %d, major length %d, last length %d, average length %d\n", qlen,pass, bodyQueryLen, lastQueryLen, (bodyQueryLen * (pass - 1) + lastQueryLen) / pass);
	}
#else
	setMultipass(qlen);
#endif
}

void KNCWorker::search(){
	__ONMIC__ __declspec(align(16)) char* 	devDBSeq  = compEntry->getBuf();
	__ONMIC__ __declspec(align(64)) int* 	deviceMap = compEntry->getMap();
	__ONMIC__ __declspec(align(64)) int* 	devResult = compEntry->getScores();
	__ONMIC__ __declspec(align(64)) int* 		deviceMatrix = plainMatrix;
	__ONMIC__ int   batchNum    = compEntry->getBatchNum();
	__ONMIC__ char* deviceQuery    = this->query;
	__ONMIC__ int 	deviceQueryLen = this->queryLen;
	__ONMIC__			int*		g_profile = _g_profile;
	__ONMIC__			int**		g_qtable  = _g_qtable;

	__ONMIC__ int	globalCounter = -1;
	/*!DONOT USE NOCOPY HERE, USE IN LENGTH(0) INSTEAD!*/
#pragma offload target(mic:id)\
	in(devDBSeq:	length(0) REUSE) \
	in(deviceMap:	length(0) REUSE)\
	in(devResult:	length(0) REUSE)\
	in(deviceMatrix:length(0) REUSE)\
	in(g_profile:   length(0) REUSE)\
	in(g_qtable:    length(0) REUSE)\
	wait(devDBSeq)
	{
#pragma omp parallel num_threads(240)
		Compute(deviceQuery, deviceQueryLen, devDBSeq, deviceMap, batchNum, devResult, deviceMatrix, globalCounter, pass, bodyQueryLen, lastQueryLen, g_profile, g_qtable);
	}
	
#pragma offload_transfer target(mic:id)\
		out(devResult:length(batchNum * 16) REUSE)
}


void KNCWorker::asyncCopy(){
	__ONMIC__ char* 	devDBSeqCopy  = buf[bufFlag];
	__ONMIC__ int* 		devMapCopy = fillMap[bufFlag];
	int 			batchNum = compEntry->getBatchNum();
#pragma offload_transfer target(mic:id)\
	in(devDBSeqCopy:	length(bufferSize) 	REUSE)\
	in(devMapCopy:		length(batchNum * 2) 	REUSE)\
	signal(devDBSeqCopy)
}


void KNCWorker::alloc(){
	buf[0] = (char*) _mm_malloc(sizeof(char) * bufferSize, 64);
	buf[1] = (char*) _mm_malloc(sizeof(char) * bufferSize, 64);

	fillMap[0] = (int *) _mm_malloc(sizeof(int) * mapSize * 2, 64);
	fillMap[1] = (int *) _mm_malloc(sizeof(int) * mapSize * 2, 64);

	score[0] = (int *) _mm_malloc(sizeof(int) * mapSize * 16, 64);
	score[1] = (int *) _mm_malloc(sizeof(int) * mapSize * 16, 64);

	_g_profile = (int*) _mm_malloc(sizeof(int) * 4 * 16 * 32 * 240, 64);
	_g_qtable  = (int**) _mm_malloc(sizeof(int*) * queryLen * 240, 64);

	entry[0].setEntry(buf[0], fillMap[0], score[0]);
	entry[1].setEntry(buf[1], fillMap[1], score[1]);

	compEntry = entry + 1;
	auxEntry = entry;

	__ONMIC__ __declspec(align(16)) char*	devBuf = buf[0];
	__ONMIC__ __declspec(align(64)) int*    devMap = fillMap[0];
	__ONMIC__ __declspec(align(64)) int*	devResult = score[0];
	__ONMIC__ __declspec(align(64)) int*	deviceMatrix = plainMatrix;
	__ONMIC__ 			char*	deviceQuery;
	__ONMIC__ 			int	deviceQueryLen;

	__ONMIC__			int*		g_profile = _g_profile;
	__ONMIC__			int**		g_qtable  = _g_qtable;

	deviceQuery = this->query;
	deviceQueryLen = this->queryLen;
#pragma offload target(mic:id)\
	in(deviceQuery:		length(deviceQueryLen) 	ALLOC)\
	in(deviceMatrix:	length(1024) 		ALLOC align(64))\
	nocopy(devMap:		length(mapSize * 2) 	ALLOC align(64))\
	nocopy(devBuf:		length(bufferSize) 	ALLOC align(16))\
	nocopy(devResult:	length(mapSize * 16) 	ALLOC align(64))\
	nocopy(g_profile:	length(4 * 16 * 32 * 240) ALLOC)\
	nocopy(g_qtable:	length(deviceQueryLen * 240) ALLOC )\
	in(deviceQueryLen, pass, bodyQueryLen, lastQueryLen)
	{
		pfalloc(deviceQuery, deviceQueryLen, 240, g_profile, g_qtable);
	}

	devBuf = buf[1];
	devMap = fillMap[1];
	devResult = score[1];
#pragma offload_transfer target(mic:id)\
	nocopy(devMap:		length(mapSize * 2) 	ALLOC align(64))\
	nocopy(devBuf:		length(bufferSize) 	ALLOC align(16))\
	nocopy(devResult:	length(mapSize * 16) 	ALLOC align(64))
}

void KNCWorker::free(){
	__ONMIC__ __declspec(align(16)) char*	devBuf = buf[0];
	__ONMIC__ __declspec(align(64)) int*      	devMap = fillMap[0];
	__ONMIC__ __declspec(align(64)) int*	devResult = score[0];
	__ONMIC__ 			char*	 	deviceQuery;
	__ONMIC__ 			int	 	deviceQueryLen;
	__ONMIC__ __declspec(align(64)) int32_t*	deviceMatrix = plainMatrix;
	__ONMIC__			int*		g_profile = _g_profile;
	__ONMIC__			int**		g_qtable  = _g_qtable;
	deviceQuery = this->query;
	deviceQueryLen = this->queryLen;

#pragma offload target(mic:id)\
	nocopy(deviceQuery:	length(deviceQueryLen) 		FREE)\
	nocopy(devBuf:		length(bufferSize)     		FREE align(16))\
	nocopy(devMap:		length(mapSize * 2) 		FREE align(64))\
	nocopy(devResult:	length(mapSize * 16) 		FREE align(64))\
	nocopy(g_profile:	length(4 * 16 * 32 * 240) 	FREE)\
	nocopy(g_qtable:	length(deviceQueryLen * 240) 	FREE)\
	nocopy(deviceMatrix: 					FREE align(64))
	;
	devBuf = buf[1];
	devMap = fillMap[1];
	devResult = score[1];
#pragma offload target(mic:id)\
	nocopy(devBuf:		length(bufferSize)  	FREE align(16))\
	nocopy(devMap:		length(mapSize * 2) 	FREE align(64))\
	nocopy(devResult:	length(mapSize * 16) 	FREE align(64))
	;
	_mm_free(fillMap[0]);
	_mm_free(fillMap[1]);
	_mm_free(score[0]);
	_mm_free(score[1]);
	_mm_free(buf[0]);
	_mm_free(buf[1]);
}


void KNCWorker::showPerformance(){
	double startTimeSec = computeStartTime.time + computeStartTime.millitm / 1000.0;
	double endTimeSec = computeStopTime.time + computeStopTime.millitm / 1000.0;
	double computeTime = endTimeSec - startTimeSec;
	printf("MIC computing time %lfs, AARes %ld, GCUPs %lf\n", computeTime, 
			TotalAminoAcidResidue, TotalAminoAcidResidue * queryLen
			/ computeTime / 1000000000);
}
