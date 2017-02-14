#ifndef CUDAWORKER_H_
#define CUDAWORKER_H_

#include "Worker.h"
#include "RecalcWorker.h"
#include "Defs.h"

#include <vector_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class CUDAWorker: public Worker
{
	public:
		CUDAWorker(DataPool* loader, Param* params, RecalcWorker* recalcWorker, int deviceId);
		~CUDAWorker();
		void setQuery(const char* query, const size_t qlen);

	private:

		void search();
		void alloc();
		void free();
		void asyncCopy();
		void showPerformance();

		RecalcWorker* recalcWorker;
		void packDB();
		
		//void launch();
		void makeQueryProfile(const char* query, const size_t qlen);

		int* score[2];

		int* hostResult;
		int* devResult;
		
		bool queryProfileFlag;

		int4 *deviceQueryProfile;

		unsigned* dbSeqPacked[2];
		unsigned* deviceBuffer[2];

		/*workload for each warp in a chunk*/
		int* chunkIndices;
		int* devChunkIndices[2];
		unsigned* deviceMap[2];
		size_t curBaseIndex;

		int4* globalArray;

		size_t globalPitch;

		const int warpSize;
		int numBlocks;
		int numThreads;
		int numWarps;
		//	const int batchSize;
		size_t maxMapSize;

		void who(){
			printf("CUDA\n");
		}
	protected:
		cudaStream_t streams[2];
		int deviceId;

	private:
		cudaArray_t cu_array;
		cudaArray_t cudaPtxQueryPrf;
		int score_limit_7;
		char* deviceQuery;
		int* deviceMatrix;
};

#endif /* CUDAWORKER_H_ */
