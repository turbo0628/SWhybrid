/*
 * RecalcWorker.h
 *
 *  Created on: 2015-3-10
 *      Author: lan
 */

#ifndef RECALCWORKER_H_
#define RECALCWORKER_H_


#include "ResultList.h"
#include "Param.h"
#include "Bucket.h"

#include <stdio.h>
#include <list>
#include <vector>
#include <sys/timeb.h>

using std::list;
using std::string;
using std::vector;

typedef char BYTE;

class RecalcWorker 
{
	public:
		RecalcWorker(Param* params);
		virtual ~RecalcWorker();

		//push
		void pushOverflowSeq(
				const BYTE* seq,
				size_t len,
				const size_t idx
				);

		void setQuery(const char* in_query, const size_t queryLen);

		void launch(){
			printf("#Recalc# symbol size %.2lfMB\n", (double) recalcSize / (double) (1<<20));
			printf("#Recalc# sequence num %ld\n", recalcNum);
			if(recalcSize != 0)
				recalc_launch();	
		}

		size_t getNum(){
			return recalcNum;
		}

		ResultList<int>& getResult(){
			return resultList;
		}

		double getStartTime(){
			return computeStartTime.millitm / 1000.0 + computeStartTime.time;
		}
		double getStopTime(){
			return computeStopTime.millitm / 1000.0 + computeStopTime.time;
		}

	private:

		timeb computeStartTime;
		timeb computeStopTime;
		int sseThreads;
		//for recalucate numbers
		size_t recalcSize;
		size_t recalcNum;
		BYTE*  recalcBuf;

		std::vector<size_t>	vec_indices;
		std::vector<int> 	vec_pos;

		void   recalc_launch();
		size_t recalcBufferSize;
		const size_t bufferSize;
		short* score_matrix_16;
		short* dup_score_matrix_16;
		char*  dup_score_matrix_8;
		BYTE* query;
		size_t queryLen;

		Param* params;

		ResultList<int> resultList;
		pthread_mutex_t pushMutex;
		void lock(){
			pthread_mutex_lock(&pushMutex);
		}

		void unlock(){
			pthread_mutex_unlock(&pushMutex);
		}

		vector<Bucket> buckets;
		vector<size_t> arrLength;
};

#endif /* RecalcWORKER_H_ */
