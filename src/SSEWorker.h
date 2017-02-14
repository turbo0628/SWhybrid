/*
 * SSEWorker.h
 *
 *  Created on: 2014-7-18
 *      Author: lan
 */

#ifndef SSEWORKER_H_
#define SSEWORKER_H_


#include "Worker.h"
#include "RecalcWorker.h"
#include <list>
#include <vector>

using std::list;
using std::string;
using std::vector;

typedef char BYTE;
typedef short WORD;

class SSEWorker : public Worker
{
public:
	SSEWorker(DataPool* loader, Param* params, RecalcWorker* recalcWorker);
	virtual ~SSEWorker();

	void who(){
			printf("SSE\n");
	}
	void setQuery(const char* query, const size_t qlen);
private:
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
	BYTE* recalcBuf;
	std::vector<size_t>	vec_indices;
	std::vector<int> 	vec_pos;

	void handleOverflow(
		BYTE* 		dbSeq,
		int* 		map,
		int		batchNum,
		size_t 		baseIndex
		);

	//for recalc
	void recalc_launch();
	short* score_matrix_16;

	int* overflow_indices;

	int* score[2];

	size_t recalcBufferSize;
};

#endif /* SSEWORKER_H_ */
