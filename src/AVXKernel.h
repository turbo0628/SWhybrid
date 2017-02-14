#ifndef AVXKERNEL_H_
#define AVXKERNEL_H_

void avxInitLocks();
void avxDestroyLocks();

void makeDupMatrix(short* dst16, char* dst8, int* src); 

void avxCompute(
		char* sseQuery,
		int   qlen,
		int  gapoe,
		int  gape,
		char* score_matrix, 
		char* deviceDBSeq, 
		int* map,
		size_t batchNum, 
		int* sseResult,
		int* overflow_indices,
		volatile int&       overflow_cnt,
		volatile int&       globalCounter,
		pthread_mutex_t&    idxMutex,
		pthread_mutex_t&    cntMutex,
		const int	    SCORE_LIMIT_7,
		char               *g_profile,
		char               **g_qtable,
		char*               externHF
		);

void avxCompute16(
		short* dup_score_matrix,
		char* dbSeq,
		int* vecPos,
		int numSeqs,
		int* result,
		int start_idx,
		char* sseQuery,
		size_t qlen
		);
#endif
