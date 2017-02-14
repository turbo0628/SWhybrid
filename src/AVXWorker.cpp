#include "AVXWorker.h"
#include "AVXKernel.h"

#ifdef HAVE_SSSE3
#include <tmmintrin.h>
#else
#include <emmintrin.h>
#endif
#if defined(__INTEL_COMPILER)
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif // defined(__GNUC__)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <sys/timeb.h>
#include <omp.h>
#include <immintrin.h>

#ifdef USE_HBW
#include <hbwmalloc.h>
#endif

using namespace std;

#define CDEPTH 4

#define BUFFERSIZE 2048
#define MAXNAMESIZE 2048

//static int overflow_cnt = 0;
static int SCORE_LIMIT_7 = 0;

static int q, r;


AVXWorker::AVXWorker(DataPool* loader, Param* params, RecalcWorker* recalcWorker):
	Worker(loader, params, recalcWorker, 1, 32,sizeof(char)),
#ifdef WITH_KNL_AVX2
				sseThreads(params->cpuNum / 4),
#else
				sseThreads(params->cpuNum),
#endif
				overflow_cnt(0),
				mapSize(4 << 20)
{
	gape = params->gapExtend;
	gapoe = params->gapOE;	
	bufferSize = 1 << 27;
	cpuFlag = true;

	//score_matrix = new char[32 * 32];
	//score_matrix_16 = new short[32 * 32];

	

	SCORE_LIMIT_7 = score_limit_7;
}

AVXWorker::~AVXWorker()
{
}

void AVXWorker::packDB(){
	Entry* pEntry = entry + !bufFlag;

	fillBatchNum = 0;
	char*	srcBuf   = dbSeqBuf;
	char*  	dstBuf   = pEntry->getBuf();
	int* 	pMap     = pEntry->getMap(); 
	batchInfo* infoData = info;
	int curPos = 0;
	int curSize = 0;

	/*copy out the buffer and transfer symbol to code*/
	this->TotalAminoAcidResidue += filledSize;

	__m256i rseq[32];
	__m256i  t8[32];
	__m256i t16[32];
	__m256i t32[32];
	__m256i t64[32];
	__m256i t128[32];

	char*	batchSrcBasep = srcBuf;
	char*	batchDstBasep = dstBuf;
	for (int idx = 0; idx < infoSize; ++idx) {
		int num = infoData[idx].numSeqs;
		int len = infoData[idx].seqLen;

		int pass = len / 32;
		int rem  = len % 32;
		int nBatches = num / batchSize;
		char *srcBasep = NULL;
		char *dstBasep = NULL;
		for (int b = 0; b < nBatches; ++b) {
#if 1
			srcBasep = batchSrcBasep + b * len * 32;
			dstBasep = batchDstBasep + b * len * 32;
			for (int j = 0; j < pass; ++j) {
				/*160 ops for transpose 32 * 32 matrix*/
				/*Naive method needs 1024 ops*/
				__m256i *vdstWorkp = ((__m256i *) dstBasep) + j * 32;
				for(int i = 0; i < 32; ++i){
					rseq[i] = _mm256_lddqu_si256((__m256i*)(srcBasep + j * 32 + i * len));

				}
#pragma unroll
				for(int i = 0; i < 32; i += 2){
					t8[i]     = _mm256_unpacklo_epi8(rseq[i], rseq[i + 1]);
					t8[i + 1] = _mm256_unpackhi_epi8(rseq[i], rseq[i + 1]);
				}

				t16[0] = _mm256_unpacklo_epi16(t8[0], t8[2]);
				t16[1] = _mm256_unpackhi_epi16(t8[0], t8[2]);
				t16[2] = _mm256_unpacklo_epi16(t8[4], t8[6]);
				t16[3] = _mm256_unpackhi_epi16(t8[4], t8[6]);
				t16[4] = _mm256_unpacklo_epi16(t8[8], t8[10]);
				t16[5] = _mm256_unpackhi_epi16(t8[8], t8[10]);
				t16[6] = _mm256_unpacklo_epi16(t8[12], t8[14]);
				t16[7] = _mm256_unpackhi_epi16(t8[12], t8[14]);

				t16[8] = _mm256_unpacklo_epi16(t8[16], t8[18]);
				t16[9] = _mm256_unpackhi_epi16(t8[16], t8[18]);
				t16[10] = _mm256_unpacklo_epi16(t8[20], t8[22]);
				t16[11] = _mm256_unpackhi_epi16(t8[20], t8[22]);
				t16[12] = _mm256_unpacklo_epi16(t8[24], t8[26]);
				t16[13] = _mm256_unpackhi_epi16(t8[24], t8[26]);
				t16[14] = _mm256_unpacklo_epi16(t8[28], t8[30]);
				t16[15] = _mm256_unpackhi_epi16(t8[28], t8[30]);

				t16[16] = _mm256_unpacklo_epi16(t8[1], t8[3]);
				t16[17] = _mm256_unpackhi_epi16(t8[1], t8[3]);
				t16[18] = _mm256_unpacklo_epi16(t8[5], t8[7]);
				t16[19] = _mm256_unpackhi_epi16(t8[5], t8[7]);
				t16[20] = _mm256_unpacklo_epi16(t8[9], t8[11]);
				t16[21] = _mm256_unpackhi_epi16(t8[9], t8[11]);
				t16[22] = _mm256_unpacklo_epi16(t8[13], t8[15]);
				t16[23] = _mm256_unpackhi_epi16(t8[13], t8[15]);

				t16[24] = _mm256_unpacklo_epi16(t8[17], t8[19]);
				t16[25] = _mm256_unpackhi_epi16(t8[17], t8[19]);
				t16[26] = _mm256_unpacklo_epi16(t8[19], t8[21]);
				t16[27] = _mm256_unpackhi_epi16(t8[19], t8[21]);
				t16[28] = _mm256_unpacklo_epi16(t8[23], t8[25]);
				t16[29] = _mm256_unpackhi_epi16(t8[23], t8[25]);
				t16[30] = _mm256_unpackhi_epi16(t8[27], t8[31]);
				t16[31] = _mm256_unpackhi_epi16(t8[27], t8[31]);
#if 0
#pragma unroll
				for(int i = 0; i < 32; i+=4){
					t32[i]     = _mm256_unpacklo_epi32(t16[i ], t16[i + 2]);
					t32[i + 1] = _mm256_unpackhi_epi32(t16[i ], t16[i + 2]);
					t32[i + 2] = _mm256_unpacklo_epi32(t16[i  + 1], t16[i + 3]);
					t32[i + 3] = _mm256_unpackhi_epi32(t16[i  + 1], t16[i + 3]);
				}
#else

				t32[0] = _mm256_unpacklo_epi32(t16[0], t16[2]);
				t32[1] = _mm256_unpackhi_epi32(t16[0], t16[2]);
				t32[2] = _mm256_unpacklo_epi32(t16[1], t16[3]);
				t32[3] = _mm256_unpackhi_epi32(t16[1], t16[3]);
				t32[4] = _mm256_unpacklo_epi32(t16[4], t16[6]);
				t32[5] = _mm256_unpackhi_epi32(t16[4], t16[6]);
				t32[6] = _mm256_unpacklo_epi32(t16[5], t16[7]);
				t32[7] = _mm256_unpackhi_epi32(t16[5], t16[7]);

				t32[8]  = _mm256_unpacklo_epi32(t16[8], t16[10]);
				t32[9]  = _mm256_unpackhi_epi32(t16[8], t16[10]);
				t32[10] = _mm256_unpacklo_epi32(t16[9], t16[11]);
				t32[11] = _mm256_unpackhi_epi32(t16[9], t16[11]);
				t32[12] = _mm256_unpacklo_epi32(t16[12], t16[14]);
				t32[13] = _mm256_unpackhi_epi32(t16[12], t16[14]);
				t32[14] = _mm256_unpacklo_epi32(t16[13], t16[15]);
				t32[15] = _mm256_unpackhi_epi32(t16[13], t16[15]);

				t32[16] = _mm256_unpacklo_epi32(t16[16], t16[18]);
				t32[17] = _mm256_unpackhi_epi32(t16[16], t16[18]);
				t32[18] = _mm256_unpacklo_epi32(t16[17], t16[19]);
				t32[19] = _mm256_unpackhi_epi32(t16[17], t16[19]);
				t32[20] = _mm256_unpacklo_epi32(t16[20], t16[22]);
				t32[21] = _mm256_unpackhi_epi32(t16[20], t16[22]);
				t32[22] = _mm256_unpacklo_epi32(t16[21], t16[23]);
				t32[23] = _mm256_unpackhi_epi32(t16[21], t16[23]);

				t32[24] = _mm256_unpacklo_epi32(t16[24], t16[26]);
				t32[25] = _mm256_unpackhi_epi32(t16[24], t16[26]);
				t32[26] = _mm256_unpacklo_epi32(t16[25], t16[27]);
				t32[27] = _mm256_unpackhi_epi32(t16[25], t16[27]);
				t32[28] = _mm256_unpacklo_epi32(t16[28], t16[30]);
				t32[29] = _mm256_unpackhi_epi32(t16[28], t16[30]);
				t32[30] = _mm256_unpacklo_epi32(t16[29], t16[31]);
				t32[31] = _mm256_unpackhi_epi32(t16[29], t16[31]);
#endif
				t64[0] = _mm256_unpacklo_epi64(t32[0], t32[4]);
				t64[1] = _mm256_unpackhi_epi64(t32[0], t32[4]);
				t64[2] = _mm256_unpacklo_epi64(t32[1], t32[5]);
				t64[3] = _mm256_unpackhi_epi64(t32[1], t32[5]);
				t64[4] = _mm256_unpacklo_epi64(t32[2], t32[6]);
				t64[5] = _mm256_unpackhi_epi64(t32[2], t32[6]);
				t64[6] = _mm256_unpacklo_epi64(t32[3], t32[7]);
				t64[7] = _mm256_unpackhi_epi64(t32[3], t32[7]);

				t64[8] = _mm256_unpacklo_epi64(t32[16], t32[20]);
				t64[9] = _mm256_unpackhi_epi64(t32[16], t32[20]);
				t64[10]= _mm256_unpacklo_epi64(t32[17], t32[21]);
				t64[11]= _mm256_unpackhi_epi64(t32[17], t32[21]);
				t64[12]= _mm256_unpacklo_epi64(t32[18], t32[22]);
				t64[13]= _mm256_unpackhi_epi64(t32[18], t32[22]);
				t64[14]= _mm256_unpacklo_epi64(t32[19], t32[23]);
				t64[15]= _mm256_unpackhi_epi64(t32[19], t32[23]);

				t64[16]= _mm256_unpacklo_epi64(t32[8], t32[12]);
				t64[17]= _mm256_unpackhi_epi64(t32[8], t32[12]);
				t64[18] = _mm256_unpacklo_epi64(t32[9], t32[13]);
				t64[19] = _mm256_unpackhi_epi64(t32[9], t32[13]);
				t64[20] = _mm256_unpacklo_epi64(t32[10], t32[14]);
				t64[21] = _mm256_unpackhi_epi64(t32[10], t32[14]);
				t64[22] = _mm256_unpacklo_epi64(t32[11], t32[15]);
				t64[23] = _mm256_unpackhi_epi64(t32[11], t32[15]);

				t64[24]= _mm256_unpacklo_epi64(t32[24], t32[28]);
				t64[25]= _mm256_unpackhi_epi64(t32[24], t32[28]);
				t64[26] = _mm256_unpacklo_epi64(t32[25], t32[29]);
				t64[27] = _mm256_unpackhi_epi64(t32[25], t32[29]);
				t64[28] = _mm256_unpacklo_epi64(t32[26], t32[30]);
				t64[29] = _mm256_unpackhi_epi64(t32[26], t32[30]);
				t64[30] = _mm256_unpacklo_epi64(t32[27], t32[31]);
				t64[31] = _mm256_unpackhi_epi64(t32[27], t32[31]);

				t128[0] = _mm256_permute2f128_si256(t64[0],  t64[16],0x20);
				t128[1] = _mm256_permute2f128_si256(t64[1],  t64[17],0x20);
				t128[2] = _mm256_permute2f128_si256(t64[2],  t64[18],0x20);
				t128[3] = _mm256_permute2f128_si256(t64[3],  t64[19],0x20);
				t128[4] = _mm256_permute2f128_si256(t64[4],  t64[20],0x20);
				t128[5] = _mm256_permute2f128_si256(t64[5],  t64[21],0x20);
				t128[6] = _mm256_permute2f128_si256(t64[6],  t64[22],0x20);
				t128[7] = _mm256_permute2f128_si256(t64[7],  t64[23],0x20);

				t128[8]  = _mm256_permute2f128_si256(t64[8],  t64[24], 0x20);
				t128[9]  = _mm256_permute2f128_si256(t64[9],  t64[25], 0x20);
				t128[10] = _mm256_permute2f128_si256(t64[10], t64[26], 0x20);
				t128[11] = _mm256_permute2f128_si256(t64[11], t64[27], 0x20);
				t128[12] = _mm256_permute2f128_si256(t64[12], t64[28], 0x20);
				t128[13] = _mm256_permute2f128_si256(t64[13], t64[29], 0x20);
				t128[14] = _mm256_permute2f128_si256(t64[14], t64[30], 0x20);
				t128[15] = _mm256_permute2f128_si256(t64[15], t64[31], 0x20);

				t128[16] = _mm256_permute2f128_si256(t64[0], t64[16], 0x31);
				t128[17] = _mm256_permute2f128_si256(t64[1], t64[17], 0x31);
				t128[18] = _mm256_permute2f128_si256(t64[2], t64[18], 0x31);
				t128[19] = _mm256_permute2f128_si256(t64[3], t64[19], 0x31);
				t128[20] = _mm256_permute2f128_si256(t64[4], t64[20], 0x31);
				t128[21] = _mm256_permute2f128_si256(t64[5], t64[21], 0x31);
				t128[22] = _mm256_permute2f128_si256(t64[6], t64[22], 0x31);
				t128[23] = _mm256_permute2f128_si256(t64[7], t64[23], 0x31);

				t128[24] = _mm256_permute2f128_si256(t64[8],  t64[24], 0x31);
				t128[25] = _mm256_permute2f128_si256(t64[9],  t64[25], 0x31);
				t128[26] = _mm256_permute2f128_si256(t64[10], t64[26], 0x31);
				t128[27] = _mm256_permute2f128_si256(t64[11], t64[27], 0x31);
				t128[28] = _mm256_permute2f128_si256(t64[12], t64[28], 0x31);
				t128[29] = _mm256_permute2f128_si256(t64[13], t64[29], 0x31);
				t128[30] = _mm256_permute2f128_si256(t64[14], t64[30], 0x31);
				t128[31] = _mm256_permute2f128_si256(t64[15], t64[31], 0x31);

				for(int i = 0; i < 32; ++i){
					_mm256_store_si256(vdstWorkp + i, t128[i]);
				}
			}
			char *srcWorkp = srcBasep + len - rem;
			char *dstWorkp = dstBasep + (len - rem) * 32;
			for(int j = 0; j < rem; ++j){
				for(int k = 0; k < 32; ++k){
					dstWorkp[k + 32 * j] = srcWorkp[j + k * len];
				}
			}	
#else
			for(int j = 0; j < len; ++j){
				for(int k = 0; k < 32; ++k){
					dstWorkp[k + 32 * j] = srcWorkp[j + k * len];
				}
			}	
			dstWorkp += 32 * len;
			srcWorkp += 32 * len;
#endif
			*(pMap++) = curPos;
			*(pMap++) = len;
			++fillBatchNum;
			curPos 	 += 32 * len;
		}
		batchSrcBasep += num * len;
		batchDstBasep += num * len;
	}
	pEntry->setBatchNum(fillBatchNum);
	pEntry->setBaseIdx(fillBaseIndex);
	pEntry->setResultSize(fillBatchNum * batchSize);

}

void AVXWorker::setQuery(const char* in_query, const size_t queryLen)
{
	char score_matrix[1024];
	cpuFlag = true;
	this->queryLen = queryLen;
	this->query = new char[queryLen];
	memcpy(this->query, in_query, sizeof(char) * queryLen);
	for( int i = 0; i < 1024; ++i){
		int sc = params->plain_matrix[i];
		score_matrix[i] = sc;
	}
	dup_score_matrix   = (char*) _mm_malloc(32 * 64 * sizeof(char) , 32);
	dup_score_matrix16 = (short*)_mm_malloc(32 * 64 * sizeof(short), 32);

	makeDupMatrix(dup_score_matrix16, dup_score_matrix, params->plain_matrix);
#ifdef WITH_KNL_AVX2
	if(queryLen < 3072)
		sseThreads /= 2;
#endif
}

void AVXWorker::alloc(){
	pthread_mutex_init(&idxMutex, NULL);
	pthread_mutex_init(&cntMutex, NULL);

	//of base class
#ifdef USE_HBW
	hbw_posix_memalign((void**)&buf[0],32, bufferSize);
	hbw_posix_memalign((void**)&buf[1],32, bufferSize);
	hbw_posix_memalign((void**)&g_qtable, 32, sseThreads * queryLen * sizeof(char*));
	hbw_posix_memalign((void**)&g_profile,32, sseThreads * 4096 * sizeof(char));
	hbw_posix_memalign((void**)&externHF,32, bufferSize * 2);
#else
	buf[0] 	= (char*) _mm_malloc(sizeof(char) * bufferSize, 32);
	buf[1] 	= (char*) _mm_malloc(sizeof(char) * bufferSize, 32);
	externHF= (char*) _mm_malloc(sizeof(char) * bufferSize * 2, 32);
	g_qtable= (char**)_mm_malloc(sizeof(char*)* sseThreads * queryLen,32);
	g_profile= (char*)_mm_malloc(sizeof(char) * sseThreads * 4096,32);
#endif
	char** qtable  = g_qtable;
	char*  profile = g_profile;
	for(int j = 0; j != this->sseThreads; ++j){
		for (int i = 0; i < queryLen; ++i){
			qtable[i] = profile + 4 * 32 * this->query[i];
		}
		profile += 4 * 32 * 32;
		qtable  += queryLen;
	}

	fillMap[0] = (int *) _mm_malloc(sizeof(int) * mapSize * 2, 32);
	fillMap[1] = (int *) _mm_malloc(sizeof(int) * mapSize * 2, 32);

	score[0] = (int *) _mm_malloc(sizeof(int) * mapSize * 32, 32);
	score[1] = (int *) _mm_malloc(sizeof(int) * mapSize * 32, 32);

	entry[0].setEntry(buf[0], fillMap[0], score[0]);
	entry[1].setEntry(buf[1], fillMap[1], score[1]);

	compEntry = entry + 1;
	auxEntry = entry;

	overflow_indices = (int *) _mm_malloc(sizeof(int) * mapSize * 32, 32);

}

void AVXWorker::free(){
	//avxDestroyLocks();
	pthread_mutex_destroy(&idxMutex);
	pthread_mutex_destroy(&cntMutex);
	_mm_free(score[0]);
	_mm_free(score[1]);
	_mm_free(overflow_indices);
	_mm_free(fillMap[0]);
	_mm_free(fillMap[1]);

	_mm_free(dup_score_matrix);
	_mm_free(dup_score_matrix16);

#ifdef USE_HBW
	hbw_free(buf[0]);
	hbw_free(buf[1]);
	hbw_free(g_profile);
	hbw_free(g_qtable);
	hbw_free(externHF);
#else
	_mm_free(buf[0]);
	_mm_free(buf[1]);
	_mm_free(g_profile);
	_mm_free(g_qtable);
	_mm_free(externHF);
#endif
}

void AVXWorker::search(){
	int globalCounter = -1;
#pragma omp parallel num_threads(sseThreads)
	avxCompute(
		this->query,
		this->queryLen,
		this->gapoe,
		this->gape,
		this->dup_score_matrix,
		compEntry->getBuf(),
		compEntry->getMap(),
		compEntry->getBatchNum(), 
		compEntry->getScores(),
		this->overflow_indices,
		this->overflow_cnt,
		globalCounter,
		this->idxMutex,
		this->cntMutex,
		SCORE_LIMIT_7,
		this->g_profile,
		this->g_qtable,
		this->externHF
	       );
	handleOverflow(
		compEntry->getBuf(),
		compEntry->getMap(),
		compEntry->getBatchNum(), 
		compEntry->getBaseIdx()
	      );

}

void AVXWorker::handleOverflow(
		char* dbSeq,
		int* map,
		int batchNum,
		size_t baseIndex
		)
{
	//printf("overflow counter %d\n", overflow_cnt);
	char* dst = recalcBuf + recalcSize;
	for(int i = 0; i != overflow_cnt; ++i)
	{
		int idx = overflow_indices[i];
		{
			int batchIdx = idx / AVX_N_LANES;
			int channel  = idx % AVX_N_LANES;
			int pos = map[batchIdx * 2];
			int len = map[batchIdx * 2 + 1];
			//printf("overflow len %d\n", len);
			char buf[len];
			char* seq = dbSeq + pos + channel;
			for(int j = 0; j != len; ++j){
				buf[j] = seq[j * AVX_N_LANES];
				if(buf[j] > 23){
					buf[j] = 23;
				}
			}
			recalcWorker->pushOverflowSeq(buf, len, idx + baseIndex);
		}
	}//for
	overflow_cnt = 0;
}

void AVXWorker::showPerformance(){
	/*show AVX performance*/
	double startTimeSec = computeStartTime.time + computeStartTime.millitm / 1000.0;
	double endTimeSec   = computeStopTime.time + computeStopTime.millitm / 1000.0;
	double computeTime  = endTimeSec - startTimeSec;
	printf("AVX computing time %lfs, AARes %ld, GCUPs %lf\n", computeTime,
			TotalAminoAcidResidue, TotalAminoAcidResidue * queryLen
			/ computeTime / 1000000000);
}
