/*
 * AVXWorker.h
 *
 *  Created on: 2016-7-1
 *      Author: lan
 */

#ifndef AVXWORKER_H_
#define AVXWORKER_H_


#include "Worker.h"
#include "RecalcWorker.h"
#include <list>
#include <vector>

using std::list;
using std::string;
using std::vector;

#define AVX_N_LANES 32

class AVXWorker : public Worker
{
public:
	AVXWorker(DataPool* loader, Param* params, RecalcWorker* recalcWorker);
	virtual ~AVXWorker();

	void who(){
			printf("AVX\n");
	}
	void setQuery(const char* query, const size_t qlen);
private:
	void packDB();
	void launch();
	void search();
	void alloc();
	void free();
	void showPerformance();
	void asyncCopy(){
		//Do nothing for host memory system
	}


private:
	const int mapSize;
	list<Overflow> overflow;
	int sseThreads;

private:
	//for recalucate numbers
	size_t recalcSize;
	size_t recalcNum;
	char* recalcBuf;
	std::vector<size_t>	  vec_indices;
	std::vector<int> vec_pos;

	void handleOverflow(
		char* 		dbSeq,
		int* 	map,
		int 	batchNum,
		size_t 		baseIndex
		);

	//for recalc
	void recalc_launch();
	short* score_matrix_16;

	int* overflow_indices;

	int* score[2];

	size_t recalcBufferSize;

	//char* score_matrix;
	char* dup_score_matrix;
	short* dup_score_matrix16;

	pthread_mutex_t idxMutex;
	pthread_mutex_t cntMutex;
	int gapoe;
	int gape;
		
	char *externHF;
	char **g_qtable;
	char *g_profile;
	int overflow_cnt;
};

#endif /* SSEWORKER_H_ */
