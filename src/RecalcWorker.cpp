/*
 * RecalcWorker.cpp
 *
 *  Created on: 2014-7-18
 *      Author: lan
 */

#include "RecalcWorker.h"


#include "AVXKernel.h"
#include "SSEKernel.h"

#include <cstdio>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <sys/timeb.h>
#include <vector>
#include <unistd.h>
#include <omp.h>
#include <mm_malloc.h>

using namespace std;
#define CDEPTH 4

#define BUFFERSIZE 2048
#define MAXNAMESIZE 2048


RecalcWorker::RecalcWorker(Param* params):
				params(params),
				sseThreads(params->cpuNum),
				recalcSize(0),
				recalcNum(0),
				bufferSize(1 << 28),
				resultList(params->topNum)
{
	recalcBufferSize = bufferSize;

	/*for 16-bit Recalc recalculate*/
	recalcBuf = (char*) malloc(sizeof(char) * bufferSize);

	score_matrix_16 = new short[32 * 32];

	dup_score_matrix_16 = (short*) _mm_malloc(sizeof(short) * 32 * 64, 32);
	dup_score_matrix_8  = (char*)  _mm_malloc(sizeof(char)  * 32 * 64, 32);

	//makeDupMatrix(dup_score_matrix_16, dup_score_matrix_8, params->plain_matrix);

	//for(int i = 0; i != 32; ++i){
		//for(int j = 0; j != 64; ++j)
			//printf("%d ", dup_score_matrix_16[i*64 + j]);
		//printf("\n");
	//}

	pthread_mutex_init(&pushMutex, NULL);

}

RecalcWorker::~RecalcWorker()
{
	pthread_mutex_destroy(&pushMutex);
	free(recalcBuf);

	_mm_free(dup_score_matrix_16);
	_mm_free(dup_score_matrix_8);
	delete [] score_matrix_16;
	delete [] query;
}

void RecalcWorker::setQuery(const char* in_query, const size_t queryLen)
{
	this->queryLen = queryLen;
	this->query = new BYTE[queryLen];
	memcpy(this->query, in_query, sizeof(char) * queryLen);
	int hi = 0;
	for (int i = 0; i < 32; ++i)
		for (int j = 0; j < 32; ++j){
			int sc = params->matrix[i][j];
			if(sc > hi)
				hi = sc;
			score_matrix_16[i * 32 + j] = (short) sc;
		}
}

static int compare_ints(const void* a, const void* b)   // comparison function
{
	int arg1 = *reinterpret_cast<const int*>(a);
	int arg2 = *reinterpret_cast<const int*>(b);
	if (arg1 < arg2)
		return 1;
	if (arg1 > arg2)
		return -1;
	return 0;
}


void RecalcWorker::recalc_launch(){
	ftime(&computeStartTime);
#if 0
	//char *buf0 = (char *) _mm_malloc(1<<27, 32);
	char   *buf1   = (char *)   _mm_malloc(1<<27, 16);
	int    *map    = (int *)    _mm_malloc(1<<16, 16);
	size_t *arrIdx = (size_t *) _mm_malloc(1<<16, 16);

	char   *pBuf1 = buf1;
	int    *pMap = map;
	size_t *pIdx = arrIdx;

	size_t batchNum = 0;

	printf("Buckets size %ld\n", buckets.size());


	string bucketBuf;
	const int batchWidth = 8;
	recalcNum = 0;
	int curPos = 0;
	for(size_t i = 0; i != buckets.size(); ++i){
		//pBuf0 = buf0;
		bucketBuf.clear();
		size_t len = buckets[i].getLen();
		size_t num = 0;
		buckets[i].flushOutPad(bucketBuf, pIdx, &num);
		pIdx += num;
		recalcNum += num;


		int pass = num / batchWidth;
		//const char *pBuf0 = bucketBuf.data();
		for (int i = 0; i < pass; ++i) {
			for (int j = 0; j < len; j++) {
				for(int k = 0; k < batchWidth; ++k){
					*(pBuf1++) = bucketBuf[j + k * len + i * batchWidth * len];
				}
			}
			*(pMap++) = curPos;
			*(pMap++) = len;
			curPos 	 += batchWidth * len;
			++batchNum;
		}
	}

	printf("batchNum %ld\n",  batchNum);
	printf("recalcNum %ld\n", recalcNum);

	int* recalcResults = (int *) _mm_malloc(sizeof(int) * recalcNum, 16);
	memset(recalcResults, 0, sizeof(int) * recalcNum);
	int globalCounter = -1;
#pragma omp parallel num_threads(1)
	ComputeRecalc(
			query, 
			queryLen, 
			12,
			2,
			score_matrix_16, 
			buf1,
			map,
			batchNum,
			recalcResults,
			globalCounter
		     );

	for(size_t i = 0; i != recalcNum; ++i)
	{
		resultList.push(recalcResults[i], arrIdx[i]);
	}
#else
#if 0
	size_t num = 0;
	for(size_t i = 0; i != recalcSize; ++i)
	{
		//printf("%d ", recalcBuf[i]);
		if(((int)recalcBuf[i]) > 23)
			printf("error #%ld size %ld\n", ++num, recalcSize);
	}
	printf("\n");
	printf("total number is %ld\n", recalcNum);
#endif
	/*launch*/
	//currently we don't care about very high scores

	int* 	recalc_result = (int *) malloc(sizeof(int) * recalcNum);
	//sseThreads = 1;

	int 	start_idx[sseThreads];
	int 	local_seqs_num[sseThreads];
	size_t 	thread_num = sseThreads;

	memset(recalc_result, 0, sizeof(int) * recalcNum);

	int N_lanes = 16;

	if (recalcNum < (size_t) sseThreads * N_lanes){
		thread_num = (recalcNum + N_lanes - 1) / N_lanes;
	}	


	/*assign work by balanced residues to avoid overhead*/
	int chunkSize = (recalcSize + thread_num - 1) / thread_num ;
	int j = 1;
	for(int i = 1; i != sseThreads; ++i){
		start_idx[i] = -1;
		local_seqs_num[i] = 0;
	}

	start_idx[0] = 0;	
	int seqCnt = 0;
	for(size_t i = 1; i != recalcNum; ++i){
		if(vec_pos[i] >= j * chunkSize){
			start_idx[j] = i;
			local_seqs_num[j - 1] = i - start_idx[j - 1];
			++j;
			++seqCnt;
		}
	}

	int rem = recalcNum - start_idx[thread_num - 1];
	if(rem == 0){
		start_idx[thread_num - 1] = -1;	
	}else{
		local_seqs_num[thread_num - 1] = rem;
	}
	


#pragma omp parallel for num_threads(thread_num)
	for(size_t i = 0; i < thread_num; ++i){
#ifdef WITH_AVX2
		avxCompute16(	score_matrix_16, 
				recalcBuf, 
				vec_pos.data(), 
				local_seqs_num[i], 
				recalc_result, 
				start_idx[i], 
				query, 
				queryLen);
#else
		Compute16(	score_matrix_16, 
				recalcBuf, 
				vec_pos.data(), 
				local_seqs_num[i], 
				recalc_result, 
				start_idx[i], 
				query, 
				queryLen);
#endif
	}//parallel for

	//save scores back
	for(size_t i = 0; i != recalcNum; ++i)
	{
		//if(recalc_result[i] >= 400)
		//printf("score %d %ld\n", recalc_result[i], i);
		resultList.push(recalc_result[i], vec_indices[i]);
	}

	/*clean*/
	recalcNum  = 0;
	recalcSize = 0;
	//vec_indices.clear();
	vec_pos.clear();
	free(recalc_result);
#endif
	ftime(&computeStopTime);
	printf("Recalculator takes %.2lfs\n", params->getTime(computeStartTime, computeStopTime));	
}

void RecalcWorker::pushOverflowSeq(const BYTE* seq, size_t len, const size_t idx)
{
	lock();

#if 1
	//realloc if the buffer is too small
	if(recalcSize + len > recalcBufferSize){
		printf("Buffer Overflow! size %ld num %ld\n", recalcSize, recalcNum);
#if 1
		recalcBufferSize += bufferSize;
		recalcBuf = (BYTE *) realloc(recalcBuf, sizeof(BYTE) * recalcBufferSize);
		if(recalcBuf == NULL){
			fprintf(stderr, "ERROR: no enough memory when realloc for recalculating buffer\n");
			exit(-1);
		}
#else
		locksse();
		recalc_launch();
		unlocksse();
#endif
		}

	BYTE* dst = recalcBuf + recalcSize;
	
	//copy sequence
	for(size_t j = 0; j < len; ++j){

		/*There's still an error in the dbMaker that '@' should be replaced by 23*/
		if(seq[j] > 23){
			//printf("warning! illegal residue %d\n", seq[j]);
			*(dst++) = 23;
			continue;
		}
		*(dst++) = seq[j];
	}
	//pad if the ending symbol is not 23
	if(*(dst-1) != 23){
		*(dst++)=23;
		*(dst++)=23;
		*(dst++)=23;
		*(dst++)=23;
		len += 4;
	}
	//build map
	vec_indices.push_back(idx);
	vec_pos.push_back(recalcSize);
	recalcSize += len;
	++recalcNum;
#else
	/*Find bucket and copy sequence*/
	//vector<size_t>::iterator iter = arrLength.find(len);
	
	
	int i = 0;
	for(i = 0; i < arrLength.size(); ++i){
		if(arrLength[i] == len)
			break;
	}
	string seqbuf(seq, len);	
	if(seqbuf[len-1] != 23){
		seqbuf.append(4, (char) 23);
	}
	len += 4;
	if(i < arrLength.size()){
		buckets[i].appendSeq(seqbuf.data(), seqbuf.size(), idx);
	}else{
		buckets.push_back(Bucket(len, 8));
		buckets[i].appendSeq(seqbuf.data(), seqbuf.size(), idx);
		arrLength.push_back(len);
	}
	recalcSize+=len;
#endif
	unlock();
}

