#include "CUDAWorkerVideoSIMD.h"
#include "CUDAKernelVideoSIMD.h"

#include <cstring>
#include <iostream>
#include <cstdio>

using namespace std;

#define CUDA_CHECK(value) {											\
	cudaError_t _m_cudaStat = value;									\
	if (_m_cudaStat != cudaSuccess) {									\
		fprintf(stderr, "Error %s at line %d in file %s\n",						\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);				\
		exit(1);											\
	} }

#define CUDA_CHECK_RETURN(value) {										\
	cudaError_t _m_cudaStat = value;									\
	if (_m_cudaStat != cudaSuccess) {									\
		fprintf(stderr, "Error %s at line %d in file %s\n",						\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);				\
		exit(1);											\
	} }
inline double getTime(timeb& start, timeb& stop)
{
		return (double) (stop.millitm - start.millitm) / 1000.0 + (stop.time - start.time);
}


CUDAWorkerVideoSIMD::CUDAWorkerVideoSIMD(DataPool* loader, Param* params, RecalcWorker* recalcWorker, int deviceId):
				Worker(loader, params, recalcWorker, 4, 128, sizeof(int4)),
				deviceQueryProfile(NULL),
				deviceBuffer(),
				deviceMap(),
				queryProfileFlag(false),
				warpSize(32),
				numBlocks(params->cudaProps->SMXCount * 8),
				numThreads(64),
				streams(),
				maxMapSize(0),
				deviceId(deviceId)
{
	//int b = 1 << 27;
	bufferSize = 1 << 27;

	numWarps = numBlocks * numThreads / 32;	
	maxMapSize = bufferSize / 16 / 128;//we hope that the mean of the subject sequences will not be below 16
}

CUDAWorkerVideoSIMD::~CUDAWorkerVideoSIMD() {
}


void CUDAWorkerVideoSIMD::alloc(){
	CUDA_CHECK_RETURN(cudaSetDevice(deviceId));
#ifdef SHOW_DETAILED_TIMER
	timeb timer;
#endif
	/*buffers for packing*/
	CUDA_CHECK(cudaMallocHost((void** ) &dbSeqPacked[0], bufferSize));
	CUDA_CHECK(cudaMallocHost((void** ) &dbSeqPacked[1], bufferSize));
#ifdef SHOW_DETAILED_TIMER
	ftime(&timer);
		printf("Allocation of host memory for GPU %d ends at %lfs\n", deviceId, params->getGlobalTime(timer));
#endif
	//sequence buffer
	CUDA_CHECK(cudaMalloc((void** ) &deviceBuffer[0], bufferSize));
	CUDA_CHECK(cudaMalloc((void** ) &deviceBuffer[1], bufferSize));
	CUDA_CHECK(cudaMalloc((void** ) &globalArray, bufferSize * 2));

	//maps
	CUDA_CHECK(cudaMallocHost((void** ) &fillMap[0], sizeof(unsigned) * maxMapSize * 2));
	CUDA_CHECK(cudaMallocHost((void** ) &fillMap[1], sizeof(unsigned) * maxMapSize * 2));

	CUDA_CHECK(cudaMalloc((void** ) &deviceMap[0], sizeof(unsigned) * maxMapSize * 2));
	CUDA_CHECK(cudaMalloc((void** ) &deviceMap[1], sizeof(unsigned) * maxMapSize * 2));

#ifdef SHOW_DETAILED_TIMER
	ftime(&timer);
	printf("Allocation of device memory for GPU %d ends at %lfs\n", deviceId, params->getGlobalTime(timer));
#endif
	//results
	CUDA_CHECK(cudaMallocHost((void** ) &score[0], sizeof(uint32_t) * maxMapSize* 128));
	CUDA_CHECK(cudaMallocHost((void** ) &score[1], sizeof(uint32_t) * maxMapSize* 128));
	CUDA_CHECK(cudaMalloc((void** ) &devResult, sizeof(int) * maxMapSize * 128));

#ifdef SHOW_DETAILED_TIMER
	ftime(&timer);
	printf("Allocation of result space of size %dMB for GPU %d ends at %lfs\n", (sizeof(int) * maxMapSize * 128) >> 20, deviceId, params->getGlobalTime(timer));
#endif
	//streams, transfer->calculate
	CUDA_CHECK(cudaStreamCreate(&streams[0]));
	CUDA_CHECK(cudaStreamCreate(&streams[1]));

	buf[0] = (char*) dbSeqPacked[0];
	buf[1] = (char*) dbSeqPacked[1];


	//Initialize entries
	entry[0].setEntry(buf[0], (int*)fillMap[0], (int*)score[0]);
	entry[1].setEntry(buf[1], (int*)fillMap[1], (int*)score[1]);
	compEntry = entry + 1;
	auxEntry = entry;

	makeQueryProfile(query, queryLen);
#ifdef SHOW_DETAILED_TIMER
	timeb  timer;
	ftime(&timer);
	printf("Allocation for GPU %d ends at %lfs\n", deviceId, params->getGlobalTime(timer));
#endif
}

void CUDAWorkerVideoSIMD::free(){
	CUDA_CHECK(cudaStreamDestroy(streams[0]));
	CUDA_CHECK(cudaStreamDestroy(streams[1]));
	CUDA_CHECK(cudaFreeHost(score[0]));
	CUDA_CHECK(cudaFreeHost(score[1]));
	CUDA_CHECK(cudaFreeHost(dbSeqPacked[0]));
	CUDA_CHECK(cudaFreeHost(dbSeqPacked[1]));
	CUDA_CHECK(cudaFreeHost(chunkIndices));
	CUDA_CHECK(cudaFree(deviceBuffer[0]));
	CUDA_CHECK(cudaFree(deviceMap[0]));
	CUDA_CHECK(cudaFree(deviceMap[1]));
	CUDA_CHECK(cudaFree(devResult));
	CUDA_CHECK(cudaFree(globalArray));
	CUDA_CHECK(cudaFree(devChunkIndices[0]));
	CUDA_CHECK(cudaFree(devChunkIndices[1]));
	CUDA_CHECK(cudaFreeHost(fillMap[0]));
	CUDA_CHECK(cudaFreeHost(fillMap[1]));

	delete this->query;
}

void CUDAWorkerVideoSIMD::setQuery(const char* query, const size_t qlen) {
	//CUDA_CHECK(cudaSetDevice(deviceId));
	if (deviceQueryProfile != NULL)
		CUDA_CHECK(cudaFree(deviceQueryProfile));
	this->queryLen = qlen;
	this->query = new char[qlen];
	memcpy(this->query, query, sizeof(char) * qlen);	
}

void CUDAWorkerVideoSIMD::makeQueryProfile(const char* query, const size_t qlen) {
#ifdef SHOW_DETAILED_TIMER
timeb timer;	
	ftime(&timer);
	printf("query profile for GPU %d begins at %lfs\n", deviceId, params->getGlobalTime(timer));
#endif
	//CUDA_CHECK_RETURN(cudaSetDevice(id));
	const int MATRIX_SIZE = 32;
	size_t prfLength = qlen >> 2;

	/*construct SIMD query profile*/
	cudaExtent volumeSize = make_cudaExtent(MATRIX_SIZE, MATRIX_SIZE,
			prfLength);
	/*allocate host memory*/
	short4* hostPtxQueryPrf = NULL;
	CUDA_CHECK_RETURN(cudaMallocHost((void **) &hostPtxQueryPrf, sizeof(short4) * volumeSize.width * volumeSize.height * volumeSize.depth));	
	/*create 3D array*/
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<short4>();
	CUDA_CHECK_RETURN(cudaMalloc3DArray(&cudaPtxQueryPrf, &channelDesc, volumeSize));

	/*fill the host array*/
	int4 index, bases;
	for (int d = 0; d < volumeSize.depth; ++d) {
		index.x = d << 2;
		bases.x = query[index.x];
		index.y = index.x + 1;
		bases.y = query[index.y];
		index.w = index.x + 2;
		bases.w = query[index.w];
		index.z = index.x + 3;
		bases.z = query[index.z];
		for (int h = 0; h < volumeSize.height; ++h) {
			short4* p = hostPtxQueryPrf
				+ d * volumeSize.height * volumeSize.width
				+ h * volumeSize.width;
			for (int w = 0; w < volumeSize.width; ++w) {
				p->x = (matrix[w][bases.x] << 8)
					| (matrix[h][bases.x] & 0x0ff);
				p->y = (matrix[w][bases.y] << 8)
					| (matrix[h][bases.y] & 0x0ff);
				p->w = (matrix[w][bases.w] << 8)
					| (matrix[h][bases.w] & 0x0ff);
				p->z = (matrix[w][bases.z] << 8)
					| (matrix[h][bases.z] & 0x0ff);
				++p;
			}
		}
	}

	/*copy data to 3D array*/
	cudaMemcpy3DParms copyParams = { 0 };
	copyParams.srcPtr = make_cudaPitchedPtr((void*) hostPtxQueryPrf,
			volumeSize.width * sizeof(short4), volumeSize.width,
			volumeSize.height);
	copyParams.dstArray = cudaPtxQueryPrf;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	CUDA_CHECK_RETURN(cudaMemcpy3D(&copyParams));
	/*release host array*/
	CUDA_CHECK_RETURN(cudaFreeHost(hostPtxQueryPrf));

	/*bind query profile for inter-task quad-lane SIMD computing*/
	bindQueryPrfVariant(cudaPtxQueryPrf);

#ifdef SHOW_DETAILED_TIMER
	ftime(&timer);
	printf("query profile for GPU %d ends at %lfs\n", deviceId, params->getGlobalTime(timer));
#endif
	queryProfileFlag = true;
}


void CUDAWorkerVideoSIMD::asyncCopy(){
	uint4* dbSeqSrc = dbSeqPacked[bufFlag];
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

void CUDAWorkerVideoSIMD::showPerformance(){
	double elapsedTime = getTime(computeStartTime, computeStopTime); 
	double gcups = queryLen * TotalAminoAcidResidue / elapsedTime / 1000000000.0;
	printf("CUDA computing time: %lf, totalAARes %ld, GCUPs %lf\n", elapsedTime,TotalAminoAcidResidue, gcups);
}


void CUDAWorkerVideoSIMD::search(){
	/*kernel launch*/
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
			globalArray
		    );

	/*copy out result*/
	CUDA_CHECK_RETURN(
			cudaMemcpy(
				compEntry->getScores(),
				devResult, 
				sizeof(unsigned) * batchNum * 128,
				cudaMemcpyDeviceToHost
				)
			);
	handleOverflow(
			(unsigned*) compEntry->getScores(),
			(uint8_t*)compEntry->getBuf(),
			(unsigned*) compEntry->getMap(),
			compEntry->getResultSize(),
			compEntry->getBaseIdx()
		      );
}


void CUDAWorkerVideoSIMD::handleOverflow(
		unsigned* resultArray,
		uint8_t* seqBuffer,
		unsigned* mapData,
		size_t resultSize,
		size_t baseIndex
		)
{
		for(int i = 0; i != resultSize; ++i){
			if(resultArray[i] >= score_limit_7){
				resultArray[i]  = 0;
				size_t batchIdx = i / 128;   //index of the int batch
				size_t seqIdxInPack= i % 128;//index of the sequence in the 128 batch
				size_t packIdx = seqIdxInPack / 4;//index of the int4 pack, 0~31
				size_t laneIdx = seqIdxInPack % 4;//index of the four lanes in a int4, 0~4
				size_t pos = mapData[batchIdx * 2];
				size_t len = mapData[batchIdx * 2 + 1];
				uint8_t* seqPack = seqBuffer + (pos + packIdx) * sizeof(int4);
			
				uint8_t* seq = ( seqPack) + laneIdx * sizeof(int);
				uint8_t buf[len];
				for(int k = 0; k < len; k += 4)
				{
					buf[k    ] = seq[0];
					buf[k + 1] = seq[1];
					buf[k + 2] = seq[2];
					buf[k + 3] = seq[3];
					seq += sizeof(int4) * 32;//we have 32 int4 packed together
				}	
				recalcWorker->pushOverflowSeq((BYTE*)buf, len, baseIndex + i);
			}
		}
}
