/*
 * Worker.cpp
 *
 *  Created on: 2014-7-2
 *      Author: lan
 */

#include "Worker.h"

#include <omp.h>

#ifndef __INTEL__COMPILER
#include <mm_malloc.h>
#endif

#include <cstdio>
#include <sys/timeb.h>
#include <cstdlib>
#include <string.h>
#include <iostream>


Worker::Worker(DataPool* loader, Param* params, RecalcWorker* recalcWorker, int packSize, int batchSize, int typeSize):
				params(params),
				dbSeqBuf(NULL),
				query(NULL),
				queryLen(0),
				infoSize(0),
				filledSize(0),
				bufferSize(0),
				topNum(params->topNum),
				loader(loader),
				matrix(),
				fillBaseIndex(-1),
				info(new batchInfo[4 << 20]),
				TotalAminoAcidResidue(0),
				bufFlag(true),
				_thread(),
				loaderFlag(false),
				auxResult(NULL),
				auxResultSize(0),
				auxThread(),
				auxBaseIndex(0),
				cpuFlag(false),
				computeStartTime(),
				computeStopTime(),
				resultList(params->topNum),
				loaderMutex(loaderMutex),
				fillBatchNum(0),
				recalcWorker(recalcWorker),
				packSize(packSize),
				batchSize(batchSize),
				typeSize(typeSize)
{
	computeStartTime.time = 0;
	computeStartTime.millitm = 0;
	computeStopTime.time = 0;
	computeStopTime.millitm = 0;
	dbBuffer = (char*) _mm_malloc(sizeof(char) * maxReaderBufferSize, 32);
	dbSeqBuf = dbBuffer;
	/*the device space/init function should be placed in the thread control function*/
	int hi = 0;
	for (int i = 0; i != 32; ++i)
		for (int j = 0; j != 32; ++j){
			int sc = params->matrix[i][j];
			this->matrix[i][j] = sc;
			if(sc > hi)
				hi = sc;
		}
	score_limit_7 = 128 - hi;
}

Worker::~Worker() {
	delete [] info;
	_mm_free(dbBuffer);
}

bool Worker::fill()
{
	//Entry* pEntry = entry + !bufFlag;
	bool ret;
	try{
		ret = loader->fillChunk(dbBuffer, info, infoSize, filledSize, bufferSize, cpuFlag, fillBaseIndex);
	}catch(const char* e){
		std::cerr<<e<<std::endl;
		exit(-1);
	}
	if(infoSize != 0)
		packDB();
	return ret;
}

#if 0
template<int M, int LANES> void pack(char* &workp, char* bufp,  int len){
	/*pack one batch*/
	/*M residues are packed together for one lane*/
	for (int j = 0; j != len; j += M) {
		/*inner loop: every sequences are interleaved*/
		for(int k = 0; k != LANES; ++k){
			for(int m = 0; m  < M; ++m){
				workp[m] = bufp[j + k * len + m];
			}
			workp += M;
		}
	}
}
#endif
void Worker::packDB(){
	//printf("Base class packing\n");
	Entry* pEntry = entry + !bufFlag;

	fillBatchNum = 0;
#if 0
	char* bufWorkp = dbSeqBuf;
	int* 	pMap     = fillMap[!bufFlag];
	char*  dstWorkp = buf[!bufFlag];
#else
	char*	bufWorkp = dbSeqBuf;
	char*  	dstWorkp = pEntry->getBuf();
	int* 	pMap     = pEntry->getMap(); 
#endif
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


			int pass = num / batchSize;
			for (int i = 0; i != pass; ++i) {
				for (int j = 0; j != len; j += packSize) {
					/*inner loop: every sequences are interleaved*/
					for(int k = 0; k != batchSize; ++k){
						for(int m = 0; m  < packSize; ++m){
							//dstWorkp[m] = bufWorkp[j + k * len + m];
							char AAsymbol = bufWorkp[j + k * len + m];
							if(AAsymbol > 23){
								//cannot distinguished symbols are labelled as -1 in the database
								AAsymbol = 23;
							}
							dstWorkp[m] = AAsymbol;
						}
						dstWorkp += packSize;
					}
				}

				*(pMap++) = curPos;
				*(pMap++) = len;
				++fillBatchNum;
				curPos 	 += batchSize * len / typeSize;
				bufWorkp += batchSize * len;
			}
	}
	pEntry->setBatchNum(fillBatchNum);
	pEntry->setBaseIdx(fillBaseIndex);
	pEntry->setResultSize(fillBatchNum * batchSize);
}

void Worker::setQuery(const char* query, const size_t qlen) {
}

void Worker::launch() {
	alloc();
	if(queryLen > 2000){
		//bufferSize /= (queryLen / 2000);
	}
	//printf("buffer %ldMB\n", bufferSize >> 20);
	loaderFlag = fill();
	if(!loaderFlag)
		return;
	swapEntries();
	asyncCopy();
	startTimer();
	while(loaderFlag){
		auxThreadStart();
		search();
		waitForAuxThread();
		swapEntries();
		asyncCopy();
	}	
	/*push the last result into the list*/
	auxWork();
	stopTimer();
	showPerformance();
	free();
}

void Worker::auxWork(){
	Entry* pEntry = auxEntry;
	if(pEntry->getResultSize() != 0){
#ifdef SHOW_CHUNK_RESULT	
		printf("------------------\n");
		who();
		printf("Result Size %d\n", auxResultSize);
		ResultList<int> tmpList(20);
		tmpList.push(pEntry->getScores(), pEntry->getResultSize(), pEntry->getBaseIdx());
		//tmpList.push(auxResult, auxResultSize, auxBaseIndex);
		tmpList.print();
		printf("------------------\n");
#endif
		//resultList.push(auxResult, auxResultSize, auxBaseIndex);
		/*push result into list*/
		resultList.push(pEntry->getScores(), pEntry->getResultSize(), pEntry->getBaseIdx());
	}
	loaderFlag = fill();
}

void Worker::alloc(){
}
void Worker::free(){
}
void Worker::asyncCopy(){
}
void Worker::search(){
}
void Worker::showPerformance(){
}

