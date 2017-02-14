/*
 * Worker.h
 *
 *  Created on: 2014-7-2
 *      Author: lan
 */

#ifndef WORKER_H_
#define WORKER_H_

#include <pthread.h>
#include <vector>
#include <cstdio>
#include <sys/timeb.h>

#include "Defs.h"
#include "Param.h"

#include <stdint.h>

#include "DataPool.h"
#include "ResultList.h"
#include "RecalcWorker.h"

class Entry
{
	public:
		Entry(){}
		Entry(char* buf, int* map, int* scores);
		~Entry(){}

		void setBatchNum(const int inBatchNum){
			this->batchNum = inBatchNum;
		}
		int getBatchNum(){
			return this->batchNum;
		}

		void setBaseIdx(const int64_t inBaseIdx){
			curBaseIdx = inBaseIdx;
		}

		size_t getBaseIdx(){
			return this->curBaseIdx;
		}

		void setEntry(char* buf, int* map, int* scores){
			this->buf = buf;
			this->scores = scores;
			this->map = map;
		}
		
		char*  getBuf(){
			return buf;
		}
		int* getMap(){
			return map;
		}
		int* getScores(){
			return scores;
		}

		void setResultSize(int resultSize){
			curScoreSz = resultSize;
		}
		int getResultSize(){
			return curScoreSz;
		}

	private:
		size_t		curBaseIdx;

		size_t 		curScoreSz;

		int 		batchNum;
		char		*buf;
		int 		*map;
		int 		mapSz;
		int		*scores;
};

class Worker
{
	public:
		Worker(DataPool* loader, Param* params, RecalcWorker* recalcWorker, int packSize, int batchSize, int typeSize);
		virtual ~Worker();

		void launch();
		virtual void alloc();
		virtual void free();
		virtual void asyncCopy();
		virtual void search();
		virtual void showPerformance();

		//virtual void packDB();
		virtual void setQuery(const char* query, const size_t qlen);

		ResultList<int>& getResult(){
			return resultList;
		}

	protected:
		bool fill();//fill a data chunk by calling the loader procedure
		bool cpuFlag;

		//for query and score matrix
		char* query;
		int queryLen;

		int matrix[32][32];
		int topNum;

		//subject database
		char* dbSeqBuf;

		ResultList<int> resultList;

		size_t fillBaseIndex;
		Param* params;

	private:
		char* dbBuffer;
		size_t* pInfoSize;
		DataPool* loader;
	protected:
		batchInfo *info;
		size_t infoSize;
		size_t filledSize;

		//timing and database stats
	public:
		double getStartTime(){
			return computeStartTime.millitm / 1000.0 + computeStartTime.time;
		}
		double getStopTime(){
			return computeStopTime.millitm / 1000.0 + computeStopTime.time;
		}	
		size_t getResidueCount(){
			return TotalAminoAcidResidue;
		}

	protected:
		int score_limit_7;
		int score_limit_16;
	protected:
		void startTimer(){ftime(&computeStartTime);}
		void stopTimer(){ftime(&computeStopTime);}
		timeb computeStartTime;
		timeb computeStopTime;

	protected:
		size_t bufferSize;
		size_t TotalAminoAcidResidue;


		//for daemon thread 
	public:
		int startThread()
		{
			int ret=pthread_create(&_thread, NULL, InternalThreadEntryFunc, this);
			return ret;
		}
		void waitForThread()
		{
			pthread_join(_thread, NULL);
		}
		void exitThread()
		{
			pthread_exit(&_thread);
		}
		void setCancelState()
		{
			pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, NULL);
		}
	private:
		void threadControl()
		{
			try
			{
				setQuery(loader->getQuery(), loader->getQueryLen());
				launch();
			} catch(const char* e){
				printf("%s", e);
			}
		}
		static void *InternalThreadEntryFunc(void *thisp)
		{
			((Worker *) thisp)->threadControl();
			((Worker *) thisp)->waitForThread();
			((Worker *) thisp)->exitThread();
			return NULL;
		}

		pthread_t _thread;
		pthread_mutex_t* loaderMutex;

		//for auxiliary thread
	protected:
		virtual void who(){}
		virtual void packDB();
		void auxWork();
		//void packDB();
		void auxThreadStart(){
			pthread_create(&auxThread, NULL, auxThreadEntryFunc, this);
		}
		void waitForAuxThread(){
			pthread_join(auxThread, NULL);
		}

		bool 	loaderFlag;

		int*      auxResult;	
		int       auxResultSize;
		size_t    auxBaseIndex;

		int* fillMap[2];
		int  fillBatchNum;

		bool bufFlag;
		char* buf[2];

		Entry entry[2];
		Entry* auxEntry;
		Entry* compEntry;
		void swapEntries(){
			bufFlag = !bufFlag;
			auxEntry = entry + !bufFlag;
			compEntry = entry + bufFlag;
		}

	private:
		static void *auxThreadEntryFunc(void *thisp)
		{
			if(thisp == 0){
				fprintf(stderr, "ERROR! Empty class pointer\n");
			}
			((Worker *) thisp)->auxWork();
			return NULL;//To avoid compiler warning
		}

		pthread_t auxThread;
	protected:
		const int batchSize;
		const int packSize;
		const int typeSize;

	protected:
		RecalcWorker* recalcWorker;
};

#endif /* WORKER_H_ */
