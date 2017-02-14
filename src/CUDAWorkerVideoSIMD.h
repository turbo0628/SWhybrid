/*
 * CUDAWorkerVideoSIMD.h
 *
 *  Created on: 2014-7-2
 *      Author: lan
 */

#ifndef CUDAWORKERVIDEOSIMD_H_
#define CUDAWORKERVIDEOSIMD_H_

#include "Worker.h"
#include "RecalcWorker.h"
#include "Defs.h"

#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class CUDAWorkerVideoSIMD: public Worker
{
	public:
		CUDAWorkerVideoSIMD(DataPool* loader, Param* params, RecalcWorker* notused, int deviceId);
		~CUDAWorkerVideoSIMD();
		void setQuery(const char* query, const size_t qlen);

	protected:
		void search();
		void alloc();
		void free();
		void showPerformance();

		void makeQueryProfile(const char* query, const size_t qlen);

		uint32_t* score[2];
		int* devResult;
		
		bool queryProfileFlag;

		int4 *deviceQueryProfile;


		uint4* dbSeqPacked[2];
		uint4* deviceBuffer[2];

		/*workload for each warp in a chunk*/
		int* chunkIndices;
		int* devChunkIndices[2];

		unsigned* deviceMap[2];
		size_t curBaseIndex;
		int4 *globalArray;

		const int warpSize;
		int numBlocks;
		int numThreads;
		int numWarps;
		//	const int batchSize;
		size_t maxMapSize;

		void who(){
			printf("CUDA Video SIMD\n");
		}

	protected:
		cudaStream_t streams[2];
		int deviceId;

	private:
		void asyncCopy();
		void handleOverflow(
				unsigned*   resultArray,
				uint8_t* seqBuffer,
				unsigned* mapData,
				size_t batchNum,
				size_t baseIndex
				);
		cudaArray_t cu_array;
		cudaArray* cudaPtxQueryPrf;
		char* deviceQuery;
		int* deviceMatrix;
};

#endif 
