#ifndef SSEKERNEL_H_
#define SSEKERNEL_H_

typedef char BYTE;
typedef short WORD;

void initLocks();
void destroyLocks();


void Compute(
		BYTE* sseQuery,
		int   qlen,
		int  gapoe,
		int  gape,
		BYTE* score_matrix, 
		BYTE* deviceDBSeq, 
		int* map,
		size_t batchNum, 
		int* sseResult,
		int* overflow_indices,
		volatile int&       overflow_cnt,
		volatile int&       globalCounter,
		const int	    SCORE_LIMIT_7
		);

void ComputeRecalc(
		char* sseQuery,
		int   qlen,
		int  gapoe,
		int  gape,
		WORD* score_matrix, 
		char* deviceDBSeq, 
		int* map,
		size_t batchNum, 
		int* sseResult,
		volatile int&       globalCounter
		);
void Compute16(
		WORD* score_matrix,
		BYTE* dbSeq,
		int* vecPos,
		int  numSeqs,
		int* result,
		int start_idx,
		BYTE* sseQuery,
		size_t qlen
		);
#endif
