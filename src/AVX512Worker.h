/*
 * AVX512Worker.h
 *
 *  Created on: 2014-7-18
 *      Author: lan
 */

#ifndef AVX512WORKER_H_
#define AVX512WORKER_H_
#include "Worker.h"

#include "RecalcWorker.h"


class AVX512Worker : public Worker
{
	public:
		AVX512Worker(DataPool* loader, Param* params, RecalcWorker* recalcWorker);
		virtual ~AVX512Worker();

		void who(){
			printf("AVX512 %d\n", id);}

		void setQuery(const char* query, const size_t qlen);

	private:
		void search();
		void alloc();
		void free();
		void asyncCopy();
		void packDB();
		void showPerformance();

	private:
		int mapSize;

	private:

		int* score[2];

		int id;

		//for score profile
		int*  _g_profile;
		int** _g_qtable;

		int *plainMatrix;
		int nThreads;
};

#endif /* AVX512WORKER_H_ */
