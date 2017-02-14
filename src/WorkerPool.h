#ifndef WORKERPOOL_H_
#define WORKERPOOL_H_

#include <pthread.h>
#include <vector>

#include "Worker.h"
#include "ResultList.h"
#include "DataPool.h"
#include "Param.h"
#include "RecalcWorker.h"

#ifdef WITH_KNL_AVX2
#include "AVXWorker.h"
#else
#ifdef WITH_KNL_AVX512
#include "AVX512Worker.h"
#else

#ifdef WITH_AVX2
#include "AVXWorker.h"
#else
#include "SSEWorker.h"
#endif

#ifdef WITH_CUDA
#include "CUDAWorkerVideoSIMD.h"
#include "CUDAWorker.h"
#endif

#ifdef WITH_KNC
#include "KNCWorker.h"
#endif

#endif
#endif

class WorkerPool {
public:
	WorkerPool(DataPool* loader, Param* parameters);
	virtual ~WorkerPool();

	void setQuery(const char* queryResidue, size_t queryLen);
private:
	void getResult();
	void showResult();
	void showPerformance();
	
	std::vector<Worker*> workerList;
	RecalcWorker* recalcWorker;

	size_t queryLen;
	int topNum;

	DataPool* loader;
	ResultList<int> resultList;
	Param* params;
};

#endif /* WORKERPOOL_H_ */
