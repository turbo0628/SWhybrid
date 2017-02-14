#include "CUDAWorker.h"
#include "CUDAKernel.h"

#include <iostream>
#include <cstdio>

#define CUDA_CHECK(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }
inline double getTime(timeb& start, timeb& stop)
{
		return (double) (stop.millitm - start.millitm) / 1000.0 + (stop.time - start.time);
}

timeb progStartTime;

CUDAWorker::CUDAWorker(DataPool* loader, Param* params, RecalcWorker* recalcWorker, int deviceId):
				Worker(loader, params, recalcWorker, 4, 32, sizeof(int)),
				deviceQueryProfile(NULL),
				deviceBuffer(),
				deviceMap(),
				queryProfileFlag(false),
				warpSize(32),
				numBlocks(24 * 10),
				numThreads(64),
				streams(),
				maxMapSize(0),
				recalcWorker(recalcWorker),
				deviceId(deviceId)
{
	ftime(&progStartTime);	
	bufferSize = 1<<28;
	numWarps = numBlocks * numThreads / 32;	
	maxMapSize = bufferSize / 16 / 32;
}

CUDAWorker::~CUDAWorker() {
}


void CUDAWorker::alloc(){
	/*buffers for packing*/
	CUDA_CHECK(cudaMallocHost((void** ) &dbSeqPacked[0], bufferSize));
	CUDA_CHECK(cudaMallocHost((void** ) &dbSeqPacked[1], bufferSize));
	//sequence buffer
	CUDA_CHECK(cudaMalloc((void** ) &deviceBuffer[0], bufferSize));
	CUDA_CHECK(cudaMalloc((void** ) &deviceBuffer[1], bufferSize));
	//maps
	CUDA_CHECK(cudaMallocHost((void** ) &fillMap[0], sizeof(unsigned) * maxMapSize * 2));
	CUDA_CHECK(cudaMallocHost((void** ) &fillMap[1], sizeof(unsigned) * maxMapSize * 2));
	CUDA_CHECK(cudaMalloc((void** ) &deviceMap[0], sizeof(unsigned) * maxMapSize * 2));
	CUDA_CHECK(cudaMalloc((void** ) &deviceMap[1], sizeof(unsigned) * maxMapSize * 2));

	//results
	CUDA_CHECK(cudaMallocHost((void** ) &score[0], sizeof(int) * maxMapSize* 32));
	CUDA_CHECK(cudaMallocHost((void** ) &score[1], sizeof(int) * maxMapSize* 32));
	CUDA_CHECK(cudaMalloc((void** ) &devResult, sizeof(int) * maxMapSize * 32));

	buf[0] = (char*) dbSeqPacked[0];
	buf[1] = (char*) dbSeqPacked[1];
	entry[0].setEntry(buf[0], (int*) fillMap[0], score[0]);
	entry[1].setEntry(buf[1], (int*) fillMap[1], score[1]);
	compEntry = entry + 1;
	auxEntry = entry;


	makeQueryProfile(query, queryLen);
	CUDA_CHECK(cudaStreamCreate(&streams[0]));
	CUDA_CHECK(cudaStreamCreate(&streams[1]));
}

void CUDAWorker::free(){
	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));
	CUDA_CHECK(cudaFreeHost(dbSeqPacked[0]));
	CUDA_CHECK(cudaFreeHost(dbSeqPacked[1]));
	CUDA_CHECK(cudaFreeHost(fillMap[0]));
	CUDA_CHECK(cudaFreeHost(fillMap[1]));
	CUDA_CHECK(cudaFree(deviceBuffer[0]));
	CUDA_CHECK(cudaFree(deviceBuffer[1]));
	CUDA_CHECK(cudaFree(deviceMap[0]));
	CUDA_CHECK(cudaFree(deviceMap[1]));
	CUDA_CHECK(cudaFreeHost(score[0]));
	CUDA_CHECK(cudaFreeHost(score[1]));
	CUDA_CHECK(cudaFree(devResult));
}

void CUDAWorker::showPerformance(){
	double elapsedTime = getTime(computeStartTime, computeStopTime); 
	double gcups = queryLen * TotalAminoAcidResidue / elapsedTime / 1000000000.0;
	printf("CUDA computing time: %lf, AARes %ld, GCUPs %lf\n", elapsedTime,TotalAminoAcidResidue, gcups);
}

void CUDAWorker::setQuery(const char* query, const size_t qlen) {
	CUDA_CHECK(cudaSetDevice(deviceId));
	if (deviceQueryProfile != NULL)
		CUDA_CHECK(cudaFree(deviceQueryProfile));
	this->queryLen = qlen;
	this->query = new char[qlen];
	memcpy(this->query, query, sizeof(char) * qlen);	
}

void CUDAWorker::makeQueryProfile(const char* query, const size_t qlen) {
	int queryLenQuad = qlen >> 2;

	int i, j;
	int4* hostQueryPrf;
		
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int4>();
	CUDA_CHECK_RETURN(cudaMallocArray(&cu_array, &channelDesc, queryLenQuad, 32));
	CUDA_CHECK_RETURN(cudaMallocHost((void **) &hostQueryPrf, sizeof(int4) * queryLenQuad * 32));

	for (i = 0; i < 32; i++) {
		int4* p = hostQueryPrf + i * queryLenQuad;
		for (j = 0; j < qlen; j += 4) {
			p->x = matrix[i][query[j]];
			p->y = matrix[i][query[j + 1]];
			p->w = matrix[i][query[j + 2]];
			p->z = matrix[i][query[j + 3]];
			p++;
		}
	}

	//set attribute of query profile
	CUDA_CHECK_RETURN(cudaMemcpy2DToArray(cu_array, 0, 0, hostQueryPrf, queryLenQuad * sizeof(int4), queryLenQuad * sizeof(int4), 32, cudaMemcpyHostToDevice));

	bindQueryPrf(cu_array);
	queryProfileFlag = true;
	CUDA_CHECK_RETURN(cudaFreeHost(hostQueryPrf));
}

void CUDAWorker::packDB(){
	//printf("CUDA packing\n");
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

	/*copy out the buffer and transfer symbol to code*/
	this->TotalAminoAcidResidue += filledSize;

		//Vectorization 
	for (int idx = 0; idx < infoSize; ++idx) {
		int num = infoData[idx].numSeqs;
		int len = infoData[idx].seqLen;
		/*limited length for intel MIC and 32-bit version CUDA kernel*/
		if(len >= 3072 * 2){
			for(int i = 0; i != num; ++i){
				recalcWorker->pushOverflowSeq((char*) bufWorkp, len, fillBaseIndex + fillBatchNum * 16 + i);
				bufWorkp += len;
			}
			continue;
		}


		int pass = num / batchSize;
		for (int i = 0; i < pass; ++i) {
			for (int j = 0; j < len; j += 4) {
				/*inner loop: every sequences are interleaved*/
				for(int k = 0; k < 32; ++k){
					*(dstWorkp++) = bufWorkp[j + k * len + 0];
					*(dstWorkp++) = bufWorkp[j + k * len + 1];
					*(dstWorkp++) = bufWorkp[j + k * len + 2];
					*(dstWorkp++) = bufWorkp[j + k * len + 3];

					//dstWorkp += packSize;
				}
			}
			*(pMap++) = curPos;
			*(pMap++) = len;
			++fillBatchNum;
			curPos 	 += batchSize * len / 4;
			bufWorkp += batchSize * len;
		}
	}
	pEntry->setBatchNum(fillBatchNum);
	pEntry->setBaseIdx(fillBaseIndex);
	pEntry->setResultSize(fillBatchNum * batchSize);
}


void CUDAWorker::asyncCopy(){
	unsigned* dbSeqSrc = dbSeqPacked[bufFlag];
	int batchNum = compEntry->getBatchNum();
	CUDA_CHECK_RETURN(
			cudaMemcpyAsync(
				deviceBuffer[bufFlag],
				dbSeqSrc,
				bufferSize, 
				cudaMemcpyHostToDevice,
				streams[bufFlag]);
			);
	CUDA_CHECK_RETURN(
			cudaMemcpyAsync(
				deviceMap[bufFlag],
				fillMap[bufFlag], 
				sizeof(unsigned) * batchNum * 2,
				cudaMemcpyHostToDevice, 
				streams[bufFlag])
			);
}
void CUDAWorker::search(){
	uint32_t batchNum = compEntry->getBatchNum();
	kernelLaunch(
			numBlocks,
			numThreads,
			streams[bufFlag],
			deviceBuffer[bufFlag],
			deviceMap[bufFlag],
			batchNum,
			queryLen,
			devResult,
			globalArray,
			globalPitch
		    );
	CUDA_CHECK_RETURN(
			cudaMemcpy(
				compEntry->getScores(),
				devResult, 
				sizeof(int) * batchNum * 32,
				cudaMemcpyDeviceToHost
				)
			);
}
