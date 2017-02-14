#include "WorkerPool.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <cstring>//for memset

using namespace std;

WorkerPool::WorkerPool(DataPool* loader, Param* params):
	params(params),
	topNum(params->topNum),
	loader(loader),
	resultList(params->topNum)
{
	cout<<"top num "<<topNum<<endl;
	if(loader == NULL){
		throw("error: no loader constructed!\n");
		exit(1);
	}

	//worker list 0 is alwasys the worker for recalculating
	recalcWorker = new RecalcWorker(params);
	recalcWorker->setQuery(loader->getQuery(), loader->getQueryLen());
#ifdef WITH_KNL_AVX2
	if(loader->getQueryLen() <= 3072){
		workerList.push_back(new AVXWorker(loader, params, recalcWorker));
		workerList.push_back(new AVXWorker(loader, params, recalcWorker));
		workerList.push_back(new AVXWorker(loader, params, recalcWorker));
		workerList.push_back(new AVXWorker(loader, params, recalcWorker));
	}else{
		workerList.push_back(new AVXWorker(loader, params, recalcWorker));
		workerList.push_back(new AVXWorker(loader, params, recalcWorker));
	}
#else
#ifdef WITH_KNL_AVX512
	if(loader->getQueryLen() <= 3072){
		workerList.push_back(new AVX512Worker(loader, params, recalcWorker));
		workerList.push_back(new AVX512Worker(loader, params, recalcWorker));
		workerList.push_back(new AVX512Worker(loader, params, recalcWorker));
		workerList.push_back(new AVX512Worker(loader, params, recalcWorker));
	}else{
		workerList.push_back(new AVX512Worker(loader, params, recalcWorker));
		workerList.push_back(new AVX512Worker(loader, params, recalcWorker));
	}
#else
#ifdef WITH_AVX2
	workerList.push_back(new AVXWorker(loader, params, recalcWorker));
#else
	workerList.push_back(new SSEWorker(loader, params, recalcWorker));
#endif


#ifdef WITH_CUDA
	for(int i = 0; i != params->cudaNum; ++i){
		if(params->cudaProps->CC == 35){
			workerList.push_back(new CUDAWorkerVideoSIMD(loader, params, recalcWorker, i));
		}else{
			workerList.push_back(new CUDAWorker(loader, params, recalcWorker, i));
		}
	}
#endif

#ifdef WITH_KNC
	for(int i = 0; i != params->micNum; ++i){
		workerList.push_back(new KNCWorker(loader, params, recalcWorker, i));
	}
#endif

#endif//KNL_AVX512
#endif//KNL_AVX2

	if(workerList.size() == 0){
		fprintf(stderr, "Device Error: No computing decive!\nPlease check build configuration\n");
	}

	for(int i = 0; i != workerList.size(); ++i)
		workerList[i]->startThread();

	params->setReservedCPUCores(workerList.size());
	queryLen = loader->getQueryLen();
}

WorkerPool::~WorkerPool()
{
	for(size_t i = 0; i != workerList.size(); ++i){
		workerList[i]->waitForThread();
	}
	recalcWorker->launch();
	showPerformance();
	getResult();	
	showResult();
	delete recalcWorker;
	for (size_t i = 0; i != workerList.size(); ++i){
		delete workerList[i];
	}
}

void WorkerPool::getResult()
{
	for(size_t i = 0; i != workerList.size(); ++i){
		resultList.mergeList(workerList[i]->getResult());
	}
	resultList.mergeList(recalcWorker->getResult());
}

void WorkerPool::showResult(){
	std::list<Result<int> >::iterator it;
	printf("----------------top %d scores---------------\n", topNum);
	for(it = resultList.getResult().begin(); it != resultList.getResult().end();++it)
	{
		printf("score %d -- %s\n", it->score, loader->getTitle(it->idx));
	}
	printf("--------------------------------------------\n");
}

void WorkerPool::setQuery(const char* queryResidue, size_t queryLen)
{
	for (size_t i = 0; i != workerList.size(); ++i)
	{
		printf("setting query\n");
		workerList[i]->setQuery(queryResidue, queryLen);
	}
	this->queryLen = queryLen;
}



void WorkerPool::showPerformance()
{
	std::vector<double> startTimeVec;
	std::vector<double> stopTimeVec;
	size_t residueCount = 0;
	residueCount = loader->getTotalResidue();
	for(size_t i = 0; i != workerList.size(); ++i)
	{
		double start = workerList[i]->getStartTime();
		double stop = workerList[i]->getStopTime();
		if(start > 0)
			startTimeVec.push_back(start);
		if(stop > 0)
			stopTimeVec.push_back(stop);
#ifdef SHOW_DETAILED_TIMER
		printf("#####start time of worker %d: %lfs\n", i, workerList[i]->getStartTime());
		printf("#####stop time of worker %d: %lfs\n", i, workerList[i]->getStopTime());
#endif
	}
	startTimeVec.push_back(recalcWorker->getStartTime());
	stopTimeVec.push_back(recalcWorker->getStopTime());
	std::sort(startTimeVec.begin(), startTimeVec.end(), std::less<double>());
	std::sort(stopTimeVec.begin(), stopTimeVec.end(), std::greater<double>());
	double time = stopTimeVec[0] - startTimeVec[0];
	double gcups = residueCount * queryLen / time / 1000000000;
	printf("total calculation time: %lf, GCUPS %lf\n, total residue %ld\n", time, gcups, residueCount);
}
