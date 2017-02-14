/*for SSE2 implementations*/
#ifdef HAVE_SSSE3
#include <tmmintrin.h>
#else
#include <emmintrin.h>
#endif


#include <omp.h>

#ifndef __INTEL_COMPILER
#include <mm_malloc.h>
#else
#include <malloc.h>
#endif

#include <immintrin.h>
#include <cstdio>
#include <string.h>
#include <unistd.h>
#include <algorithm>
#include <sys/timeb.h>
#include <vector>
#include <unistd.h>
#include <unistd.h>
#include <pthread.h>

#include "AVXKernel.h"

#ifdef USE_HBW
#include <hbwmalloc.h>
#endif

#define N_CHANNELS		32
#define CDEPTH 			4


//static pthread_mutex_t idxMutex;
//static pthread_mutex_t cntMutex;

void makeDupMatrix(short* dupMatrix16, char* dupMatrix8, int* originMatrix){
	char origin8[32 * 32];
	short origin16[32 * 32];
	for( int i = 0; i < 1024; ++i){
		origin8[i]  = (char)  originMatrix[i];
		origin16[i] = (short) originMatrix[i];
	}

	for(int i = 0; i != 64; ++i){
		memcpy(dupMatrix8  + i * 32     , origin8  + i * 16, 16 * sizeof(char ));
		memcpy(dupMatrix8  + i * 32 + 16, origin8  + i * 16, 16 * sizeof(char ));
	}
	for(int i = 0; i != 128; ++i){
		memcpy(dupMatrix16 + i * 16     , origin16 + i *  8, 8 * sizeof(short));
		memcpy(dupMatrix16 + i * 16 +  8, origin16 + i *  8, 8 * sizeof(short));
	}
}


void avxCalcScoreProfile(char* dprofile, 
		char* dup_score_matrix, 
		char* dseq_byte){

	__m256i *dseq = (__m256i*) dseq_byte;
	__m256i x, y, a, b, c, d, e, f, g, h;
	__m256i	t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13;
	__m256i	u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13;

#if 0
	__m256i	s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13;
	__m256i	v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13;
#endif

	__m256i	m0, m1, m2, m3, m4, m5, m6, m7;
	__m256i	n0, n1, n2, n3, n4, n5, n6, n7;

	x = _mm256_set1_epi8(0x10);
	y = _mm256_set1_epi8(0x80);


	//load seq
	//mark the indices larger than 16
	//shift 3 bits left to meet y
	//make the mask
	//use the mask to generate two vectors hold 
	//indices smaller/larger than 16
	
	a  = _mm256_load_si256(dseq);			
	t0 = _mm256_and_si256(a, x);
	t1 = _mm256_slli_epi16(t0, 3);
	t2 = _mm256_xor_si256(t1, y);
	m0 = _mm256_or_si256(a, t1);
	m1 = _mm256_or_si256(a, t2);

	b  = _mm256_load_si256(dseq + 1);			
	t3 = _mm256_and_si256(b, x);
	t4 = _mm256_slli_epi16(t3, 3);
	t5 = _mm256_xor_si256(t4, y);
	m2 = _mm256_or_si256(b, t4);
	m3 = _mm256_or_si256(b, t5);

	c  = _mm256_load_si256(dseq + 2);			
	u0 = _mm256_and_si256(c, x);
	u1 = _mm256_slli_epi16(u0, 3);
	u2 = _mm256_xor_si256(u1, y);
	m4 = _mm256_or_si256(c, u1);
	m5 = _mm256_or_si256(c, u2);

	d  = _mm256_load_si256(dseq + 3);			
	u3 = _mm256_and_si256(d, x);
	u4 = _mm256_slli_epi16(u3, 3);
	u5 = _mm256_xor_si256(u4, y);
	m6 = _mm256_or_si256(d, u4);
	m7 = _mm256_or_si256(d, u5);
#if 0
	/////////////////////////////////
	e  = _mm256_load_si256(dseq + 4);			
	s0 = _mm256_and_si256(e, x);
	s1 = _mm256_slli_epi16(s0, 3);
	s2 = _mm256_xor_si256(s1, y);
	n0 = _mm256_or_si256(e, s1);
	n1 = _mm256_or_si256(e, s2);

	f  = _mm256_load_si256(dseq + 5);			
	s3 = _mm256_and_si256(f, x);
	s4 = _mm256_slli_epi16(s3, 3);
	s5 = _mm256_xor_si256(s4, y);
	n2 = _mm256_or_si256(b, s4);
	n3 = _mm256_or_si256(b, s5);

	g  = _mm256_load_si256(dseq + 6);			
	v0 = _mm256_and_si256(g, x);
	v1 = _mm256_slli_epi16(v0, 3);
	v2 = _mm256_xor_si256(v1, y);
	n4 = _mm256_or_si256(g, v1);
	n5 = _mm256_or_si256(g, v2);

	h  = _mm256_load_si256(dseq + 7);			
	v3 = _mm256_and_si256(h, x);
	v4 = _mm256_slli_epi16(v3, 3);
	v5 = _mm256_xor_si256(v4, y);
	n6 = _mm256_or_si256(h, v4);
	n7 = _mm256_or_si256(h, v5);
	/////////////////////////////
#endif
#pragma unroll
	for(int i = 0; i < 24; ++i){
		t6 = _mm256_load_si256((__m256i*) (dup_score_matrix) + 2 * i);
		t7 = _mm256_load_si256((__m256i*) (dup_score_matrix) + 2 * i + 1);
		t8 = _mm256_shuffle_epi8(t6, m0);
		t9 = _mm256_shuffle_epi8(t7, m1);
		t10 = _mm256_shuffle_epi8(t6, m2);
		t11 = _mm256_shuffle_epi8(t7, m3);

		u8 = _mm256_shuffle_epi8(t6, m4);
		u9 = _mm256_shuffle_epi8(t7, m5);
		u10 = _mm256_shuffle_epi8(t6, m6);
		u11 = _mm256_shuffle_epi8(t7, m7);

		t12 = _mm256_or_si256(t8, t9);	
		t13 = _mm256_or_si256(t10, t11);	

		u12 = _mm256_or_si256(u8, u9);	
		u13 = _mm256_or_si256(u10, u11);	
		_mm256_store_si256((__m256i*)(dprofile)+4*i,  t12);
		_mm256_store_si256((__m256i*)(dprofile)+4*i+1,t13);
		_mm256_store_si256((__m256i*)(dprofile)+4*i+2,u12);
		_mm256_store_si256((__m256i*)(dprofile)+4*i+3,u13);
#if 0
		//////////////////////////////////
		s8  = _mm256_shuffle_epi8(s6, n0);
		s9  = _mm256_shuffle_epi8(s7, n1);
		s10 = _mm256_shuffle_epi8(s6, n2);
		s11 = _mm256_shuffle_epi8(s7, n3);

		v8  = _mm256_shuffle_epi8(v6, n4);
		v9  = _mm256_shuffle_epi8(v7, n5);
		v10 = _mm256_shuffle_epi8(v6, n6);
		v11 = _mm256_shuffle_epi8(v7, n7);

		s12 = _mm256_or_si256(s8,  s9);	
		s13 = _mm256_or_si256(s10, s11);	

		v12 = _mm256_or_si256(v8,  v9);	
		v13 = _mm256_or_si256(v10, v11);	
		_mm256_store_si256((__m256i*)(dprofile)+8*i+4,s12);
		_mm256_store_si256((__m256i*)(dprofile)+8*i+5,s13);
		_mm256_store_si256((__m256i*)(dprofile)+8*i+6,v12);
		_mm256_store_si256((__m256i*)(dprofile)+8*i+7,v13);
		/////////////////////////////////////
#endif

	}
}

inline void ONE_CELL_UPDATE(
		__m256i& vH,
		__m256i& vN,
		__m256i& vE,
		__m256i& vF,
		__m256i& vS,
		__m256i& vP,
		__m256i& R,
		__m256i& Q
		)
{

	vH = _mm256_adds_epi8(vH, vP);//H = H + p[q]
	vH = _mm256_max_epi8(vH, vF);//H = max(H, F)
	vH = _mm256_max_epi8(vH, vE);//H = max(H, E)
	vS = _mm256_max_epi8(vS, vH);//S = max(S, H)
	vF = _mm256_subs_epi8(vF, R);//F = F - R
	vE = _mm256_subs_epi8(vE, R);//E = E - R
	vN = vH;
	vH = _mm256_subs_epi8(vH, Q);//H = H - Q
	vE = _mm256_max_epi8(vH, vE);// E = max(H, E)
	vF = _mm256_max_epi8(vH, vF);//F = ax(H, F)
}
//#define CDEPTH 4
void avxSearchTile(__m256i *S, 
	       __m256i *hep, 
	       __m256i ** q_start, 
	       __m256i Q, 
	       __m256i R, 
	       int qlen, 
	       __m256i *Zm){

	register __m256i vZero, vE, vP, vS, vHload;
	register __m256i vH[4], vN[4], vF[4];
	//register __m256i vH[CDEPTH], vN[CDEPTH], vF[CDEPTH];
	int i, j;


	__m256i *savedH = hep;
	__m256i *savedE = hep + qlen;


	vZero = _mm256_set1_epi8(0x80);
	vE = _mm256_load_si256(savedE);
	vS = _mm256_load_si256(S);

	/*
	 * Calculation for the first row in the tile
	 */
	for (i = 0; i < 4; i++)
	{
		vH[i] = vZero;
		vF[i] = vZero;
		//loading sub score
		vP = _mm256_load_si256((q_start[0] + i));
		ONE_CELL_UPDATE(vH[i], vN[i], vE, vF[i], vS, vP, R, Q);
	}
	_mm256_store_si256(savedE, vE);
	//Calculate along the query.
	for (j = 1; j < qlen; j++)
	{
		//1st cell in the row
		//load H and then update
		vHload = _mm256_load_si256(savedH + j - 1);
		_mm256_store_si256(savedH + (j - 1), vN[3]);
		vE = _mm256_load_si256(savedE + j);
		vP = _mm256_load_si256(q_start[j]);
		ONE_CELL_UPDATE(vHload, vH[0], vE, vF[0], vS, vP, R, Q);
		//update remaining 7 cells in the row
		for (i = 1; i < 4; i++)
		{
			vP = _mm256_load_si256(q_start[j] + i);
			ONE_CELL_UPDATE(vN[i-1], vH[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm256_store_si256(savedE + j, vE);
		++j;
		if (j == qlen)
			break;
		//swap vH and vN
		vHload = _mm256_load_si256(savedH + (j - 1));
		_mm256_store_si256(savedH + (j - 1), vH[3]);
		vE = _mm256_load_si256(savedE + j);
		vP = _mm256_load_si256(q_start[j]);
		ONE_CELL_UPDATE(vHload, vN[0], vE, vF[0], vS, vP, R, Q);
		for (i = 1; i < 4; i++)
		{
			vP = _mm256_load_si256(q_start[j] + i);
			ONE_CELL_UPDATE(vH[i-1], vN[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm256_store_si256(savedE + j, vE);
	}
	_mm256_store_si256(S, vS);

}

void avxSearchTileMultiPass(__m256i *S, 
	       __m256i *hep, 
	       __m256i *backhf,
	       __m256i ** q_start, 
	       __m256i Q, 
	       __m256i R, 
	       int qlen, 
	       int startPos,
	       __m256i *Zm){

	register __m256i vZero, vE, vP[4], vS, vHload;
	register __m256i vH[4], vN[4], vF[4];
	//register __m256i vH[CDEPTH], vN[CDEPTH], vF[CDEPTH];
	int i, j;
	q_start += startPos;


	__m256i *savedH = hep;
	__m256i *savedE = hep + qlen;


	vZero = _mm256_set1_epi8(0x80);
	vE = _mm256_load_si256(savedE);
	vS = _mm256_load_si256(S);

	if(startPos == 0){
#pragma unroll
		for (i = 0; i < 4; i++){
			vH[i] = vZero;
			vF[i] = vZero;
			//loading sub score
			vP[i] = _mm256_load_si256((q_start[0] + i));
			ONE_CELL_UPDATE(vH[i], vN[i], vE, vF[i], vS, vP[i], R, Q);
		}
	}else{
		vH[0] = _mm256_load_si256(backhf + 0);
		vF[0] = _mm256_load_si256(backhf + 1);
		vH[1] = _mm256_load_si256(backhf + 2);
		vF[1] = _mm256_load_si256(backhf + 3);
		vH[2] = _mm256_load_si256(backhf + 4);
		vF[2] = _mm256_load_si256(backhf + 5);
		vH[3] = _mm256_load_si256(backhf + 6);
		vF[3] = _mm256_load_si256(backhf + 7);
		vHload = _mm256_load_si256(hep);
		_mm256_store_si256(hep, vH[3]);
		vP[0] = _mm256_load_si256((q_start[0] + 0));
		vP[1] = _mm256_load_si256((q_start[0] + 1));
		vP[2] = _mm256_load_si256((q_start[0] + 2));
		vP[3] = _mm256_load_si256((q_start[0] + 3));
		ONE_CELL_UPDATE(vHload, vN[0], vE, vF[0], vS, vP[0], R, Q);
		ONE_CELL_UPDATE(vH[0],  vN[1], vE, vF[1], vS, vP[1], R, Q);
		ONE_CELL_UPDATE(vH[1],  vN[2], vE, vF[2], vS, vP[2], R, Q);
		ONE_CELL_UPDATE(vH[2],  vN[3], vE, vF[3], vS, vP[3], R, Q);
		
	}
	_mm256_store_si256(savedE, vE);
	//Calculate along the query.
	for (j = 1; j < qlen; j++)
	{
		//1st cell in the row
		//load H and then update
		vHload = _mm256_load_si256(savedH + j - 1);
		_mm256_store_si256(savedH + (j - 1), vN[3]);
		vE = _mm256_load_si256(savedE + j);
		vP[0] = _mm256_load_si256(q_start[j]);
		vP[1] = _mm256_load_si256(q_start[j] + 1);
		vP[2] = _mm256_load_si256(q_start[j] + 2);
		vP[3] = _mm256_load_si256(q_start[j] + 3);
		ONE_CELL_UPDATE(vHload, vH[0], vE, vF[0], vS, vP[0], R, Q);
		ONE_CELL_UPDATE(vN[0], vH[1], vE, vF[1], vS, vP[1], R, Q);
		ONE_CELL_UPDATE(vN[1], vH[2], vE, vF[2], vS, vP[2], R, Q);
		ONE_CELL_UPDATE(vN[2], vH[3], vE, vF[3], vS, vP[3], R, Q);

		_mm256_store_si256(savedE + j, vE);
		++j;
		if (j == qlen)
			break;

		//swap vH and vN
		vHload = _mm256_load_si256(savedH + (j - 1));
		_mm256_store_si256(savedH + (j - 1), vH[3]);
		vE = _mm256_load_si256(savedE + j);
		vP[0] = _mm256_load_si256(q_start[j]);
		vP[1] = _mm256_load_si256(q_start[j] + 1);
		vP[2] = _mm256_load_si256(q_start[j] + 2);
		vP[3] = _mm256_load_si256(q_start[j] + 3);
		ONE_CELL_UPDATE(vHload, vN[0], vE, vF[0], vS, vP[0], R, Q);
		ONE_CELL_UPDATE(vH[0], vN[1], vE, vF[1], vS, vP[1], R, Q);
		ONE_CELL_UPDATE(vH[1], vN[2], vE, vF[2], vS, vP[2], R, Q);
		ONE_CELL_UPDATE(vH[2], vN[3], vE, vF[3], vS, vP[3], R, Q);
		_mm256_store_si256(savedE + j, vE);
	}

	_mm256_store_si256(S, vS);
	if(qlen % 2 == 0){
		_mm256_store_si256(backhf + 0, vN[0]);
		_mm256_store_si256(backhf + 1, vF[0]);
		_mm256_store_si256(backhf + 2, vN[1]);
		_mm256_store_si256(backhf + 3, vF[1]);
		_mm256_store_si256(backhf + 4, vN[2]);
		_mm256_store_si256(backhf + 5, vF[2]);
		_mm256_store_si256(backhf + 6, vN[3]);
		_mm256_store_si256(backhf + 7, vF[3]);
	}else{

		_mm256_store_si256(backhf + 0, vH[0]);
		_mm256_store_si256(backhf + 1, vF[0]);
		_mm256_store_si256(backhf + 2, vH[1]);
		_mm256_store_si256(backhf + 3, vF[1]);
		_mm256_store_si256(backhf + 4, vH[2]);
		_mm256_store_si256(backhf + 5, vF[2]);
		_mm256_store_si256(backhf + 6, vH[3]);
		_mm256_store_si256(backhf + 7, vF[3]);
	}

}

void avxCompute(
		char* query,
		int   qlen,
		int  gapoe,
		int  gape,
		char* score_matrix, 
		char* deviceDBSeq, 
		int* map,
		size_t batchNum, 
		int* result,
		int* overflow_indices,
		volatile int&       overflow_cnt,
		volatile int&       globalCounter,
		pthread_mutex_t&    idxMutex,
		pthread_mutex_t&    cntMutex,
		const int SCORE_LIMIT_7,
		char   *g_profile,
		char   **g_qtable,
		char* externHF
		)
{
	int tid = omp_get_thread_num();
	int localIdx = -1;
	int localOverflowIdx = 0;
	
	pthread_mutex_lock(&idxMutex);
	localIdx = ++globalCounter;
	pthread_mutex_unlock(&idxMutex);

	char gap_open_penalty = gapoe;
	char gap_extend_penalty = gape;

	char *dprofile = g_profile + tid * 4096;
	char **qtable  = g_qtable  + tid * qlen;

	__m256i S, Q, R, Z;
	__m256i *hep, **qp;


	Z  = _mm256_set1_epi8(0x80);
	Q  = _mm256_set1_epi8(gap_open_penalty);
	R  = _mm256_set1_epi8(gap_extend_penalty);

	const int mlen = 4096; 

#ifdef WITH_KNL_AVX2
	if(qlen < mlen){
#endif
		char *hearray = (char *)  _mm_malloc(qlen * 2 * 32, 32);
		hep = (__m256i *) hearray;
		qp = (__m256i **) qtable;
		int* scores;
		while (localIdx < (int) batchNum && localIdx > -1)
		{
			//one batch at a time
			S = Z;
			memset(hearray, 0x80, qlen * 2 * 32);
			scores = result + localIdx * 32;
			int pos = map[localIdx * 2];
			int dblen = map[localIdx * 2 + 1];
			char* dbseq = (char*)deviceDBSeq + pos;
			for (size_t i = 0; i < dblen >> 2; ++i)
			{
				avxCalcScoreProfile(dprofile, score_matrix, (char*)(dbseq + i * 32 * 4));
				avxSearchTile(&S, hep, qp, Q, R, qlen, &Z);
			}

			for(int i = 0; i != 32; ++i){
				int score = ((unsigned char*) &S)[i] ^ 0x80;
				if (score >= SCORE_LIMIT_7) {
					pthread_mutex_lock(&cntMutex);
					localOverflowIdx = overflow_cnt++;
					pthread_mutex_unlock(&cntMutex);
					overflow_indices[localOverflowIdx] = localIdx * 32 + i;
					score = 0;
				}
				scores[i] = score;
			}

			pthread_mutex_lock(&idxMutex);
			localIdx = ++globalCounter;
			pthread_mutex_unlock(&idxMutex);
		}
		_mm_free(hearray);
#ifdef WITH_KNL_AVX2
	}else{
		int pass, bodylen, lastlen;
		pass = (qlen + mlen - 1) / mlen;
		bodylen = (qlen + pass - 1) / pass;
		lastlen = qlen - bodylen * (pass - 1);
		__declspec(align(32)) char hearray[bodylen * 2 * 32];
		qp  = (__m256i **) qtable;
		hep = (__m256i *) hearray;

		int* scores;

		while (localIdx < (int) batchNum && localIdx > -1)
		{
			//one batch at a time
			S = Z;
			memset(hearray, 0x80, bodylen * 2 * 32);
			scores = result + localIdx * 32;
			int pos = map[localIdx * 2];
			int dblen = map[localIdx * 2 + 1];
			char* dbseq = (char*)deviceDBSeq + pos;
			__m256i *backhf = (__m256i *) (externHF + pos * 2);
			int qseglen = 0;
			int startPos = 0;
			for (int j = 0; j < pass; ++j){
				memset(hearray, 0x80, bodylen * 2 * 32);
				if(j == pass - 1){
					qseglen = lastlen;
				}else{
					qseglen = bodylen;
				}
				for (size_t i = 0; i < dblen >> 2; ++i)
				{
					avxCalcScoreProfile(dprofile, score_matrix, (char*)(dbseq + i * 32 * 4));
					avxSearchTileMultiPass(&S, hep, backhf + i * 8, qp, Q, R, qseglen, startPos, &Z);
				}
				startPos += qseglen;
			}
			//TODO: move it to aux thread
			//overflow detection
			for(int i = 0; i != 32; ++i){
				int score = ((unsigned char*) &S)[i] ^ 0x80;
				if (score >= SCORE_LIMIT_7) {
					pthread_mutex_lock(&cntMutex);
					localOverflowIdx = overflow_cnt++;
					pthread_mutex_unlock(&cntMutex);

					overflow_indices[localOverflowIdx] = localIdx * 32 + i;
					score = 0;
				}
				scores[i] = score;
			}

			pthread_mutex_lock(&idxMutex);
			localIdx = ++globalCounter;
			pthread_mutex_unlock(&idxMutex);

		}


	}
#endif
}

inline void ONE_CELL_UPDATE_16(
		__m256i& vH,
		__m256i& vN,
		__m256i& vE,
		__m256i& vF,
		__m256i& vS,
		__m256i& vP,
		__m256i& R,
		__m256i& Q
		)
{
	vH = _mm256_adds_epi16(vH, vP);//H = H + p[q]
	vH = _mm256_max_epi16(vH, vF);//H = max(H, F)
	vH = _mm256_max_epi16(vH, vE);//H = max(H, E)
	vS = _mm256_max_epi16(vS, vH);//S = max(S, H)
	vF = _mm256_subs_epi16(vF, R);//F = F - R
	vE = _mm256_subs_epi16(vE, R);//E = E - R
	vN = vH;
	vH = _mm256_subs_epi16(vH, Q);//H = H - Q
	vE = _mm256_max_epi16(vH, vE);// E = max(H, E)
	vF = _mm256_max_epi16(vH, vF);//F = ax(H, F)
}

inline void donormal16(
		__m256i * Sm, /* r9  */
		__m256i * hep, /* rdi */
		__m256i ** qp, /* rsi */
		__m256i * Qm, /* rdx */
		__m256i * Rm, /* rcx */
		long ql, /* r8  */
		__m256i * Zm)
{
	register __m256i vZero, vE, vP, vS, vHload;
	register __m256i vH[4], vN[4], vF[4];
	int i, j;

	__m256i **q_start = qp;
	__m256i *savedH = hep;
	__m256i *savedE = hep + ql;

	__m256i R = _mm256_load_si256(Rm);
	__m256i Q = _mm256_load_si256(Qm);


	vZero = _mm256_set1_epi16(0x8000);
	vE = _mm256_load_si256(savedE);
	vS = _mm256_load_si256(Sm);

	/*
	 * Calculation for the first row in the tile
	 */
	for (i = 0; i < 4; i++)
	{
		vH[i] = vZero;
		vF[i] = vZero;
		vP = _mm256_load_si256((q_start[0] + i));
		ONE_CELL_UPDATE_16(vH[i], vN[i], vE, vF[i], vS, vP, R, Q);
	}
	_mm256_store_si256(savedE, vE);
	//Calculate along the query.
	for (j = 1; j < ql; j++)
	{
		//1st cell in the row
		//load H and then update
		vHload = _mm256_load_si256(savedH + j - 1);
		_mm256_store_si256(savedH + (j - 1), vN[3]);
		vE = _mm256_load_si256(savedE + j);
		vP = _mm256_load_si256(q_start[j]);
		ONE_CELL_UPDATE_16(vHload, vH[0], vE, vF[0], vS, vP, R, Q);
		//update remaining 3 cells in the row
		for (i = 1; i < 4; i++)
		{
			vP = _mm256_load_si256(q_start[j] + i);
			ONE_CELL_UPDATE_16(vN[i-1], vH[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm256_store_si256(savedE + j, vE);
		++j;
		if (j == ql)
			break;
		//swap vH and vN
		vHload = _mm256_load_si256(savedH + (j - 1));
		_mm256_store_si256(savedH + (j - 1), vH[3]);
		vE = _mm256_load_si256(savedE + j);
		vP = _mm256_load_si256(q_start[j]);
		ONE_CELL_UPDATE_16(vHload, vN[0], vE, vF[0], vS, vP, R, Q);
		for (i = 1; i < 4; i++)
		{
			vP = _mm256_load_si256(q_start[j] + i);
			ONE_CELL_UPDATE_16(vH[i-1], vN[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm256_store_si256(savedE + j, vE);
	}
	_mm256_store_si256(Sm, vS);
}

inline void domasked16(__m256i * Sm, __m256i * hep, __m256i ** qp,
		__m256i * Qm, __m256i * Rm, long ql, __m256i * Zm, __m256i * Mm)
{
	register __m256i vZero, vE, vP, vS, vHload;
	register __m256i vH[4], vN[4], vF[4];
	int i, j;

	__m256i **q_start = qp;
	__m256i *savedH = hep;
	__m256i *savedE = hep + ql;

	__m256i R = _mm256_load_si256(Rm);
	__m256i Q = _mm256_load_si256(Qm);
	__m256i M = _mm256_load_si256(Mm);


	vZero = _mm256_set1_epi16(0x8000);
	vE = _mm256_load_si256(savedE);
	vE = _mm256_adds_epi16(vE, M);
	vS = _mm256_load_si256(Sm);
	vS = _mm256_adds_epi16(vS, M);

	/*
	 * Calculation for the first row in the tile
	 */
	for (i = 0; i < 4; i++)
	{
		vH[i] = vZero;
		vF[i] = vZero;
		//loading sub score
		vP = _mm256_load_si256((q_start[0] + i));
		ONE_CELL_UPDATE_16(vH[i], vN[i], vE, vF[i], vS, vP, R, Q);
	}
	_mm256_store_si256(savedE, vE);
	//Calculate along the query.
	for (j = 1; j < ql; j++)
	{
		//1st cell in the row
		//load H and then update
		vHload = _mm256_load_si256(savedH + j - 1);
		vHload = _mm256_adds_epi16(vHload, M);
		_mm256_store_si256(savedH + (j - 1), vN[3]);
		vE = _mm256_load_si256(savedE + j);
		vE = _mm256_adds_epi16(vE, M);
		vP = _mm256_load_si256(q_start[j]);
		ONE_CELL_UPDATE_16(vHload, vH[0], vE, vF[0], vS, vP, R, Q);
		//update remaining 3 cells in the row
		for (i = 1; i < 4; i++)
		{
			vP = _mm256_load_si256(q_start[j] + i);
			ONE_CELL_UPDATE_16(vN[i-1], vH[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm256_store_si256(savedE + j, vE);
		++j;
		if (j == ql)
			break;
		//swap vH and vN
		vHload = _mm256_load_si256(savedH + (j - 1));
		vHload = _mm256_adds_epi16(vHload, M);
		_mm256_store_si256(savedH + (j - 1), vH[3]);
		vE = _mm256_load_si256(savedE + j);
		vE = _mm256_adds_epi16(vE, M);
		vP = _mm256_load_si256(q_start[j]);
		ONE_CELL_UPDATE_16(vHload, vN[0], vE, vF[0], vS, vP, R, Q);
		for (i = 1; i < 4; i++)
		{
			vP = _mm256_load_si256(q_start[j] + i);
			ONE_CELL_UPDATE_16(vH[i-1], vN[i], vE, vF[i], vS, vP, R, Q);
		}
		_mm256_store_si256(savedE + j, vE);
	}
	_mm256_store_si256(Sm, vS);
}

#define N16_CHANNELS 16
//#define CDEPTH       4

inline void dprofile_fill16(short * dprofile_word, short * dup_score_matrix_word,
			char * dseq)
{
#if 0
	__m256i xmm0,  xmm1,  xmm2,  xmm3,  xmm4,  xmm5,  xmm6,  xmm7;
	__m256i xmm8,  xmm9,  xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;
	__m256i xmm16, xmm17, xmm18, xmm19, xmm20, xmm21, xmm22, xmm23;
	__m256i xmm24, xmm25, xmm26, xmm27, xmm28, xmm29, xmm30, xmm31;

	__m256i *dprofile = (__m256i *)dprofile_word;

	for (int j = 0; j < CDEPTH; j++)
	{
		int d[N16_CHANNELS];
		for (int z = 0; z < N16_CHANNELS; z++)
			d[z] = dseq[j * N16_CHANNELS + z] << 5;

		for (int i = 0; i < 32; i += 16)
		{

			xmm0 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[0] + i));
			xmm1 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[1] + i));
			xmm2 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[2] + i));
			xmm3 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[3] + i));
			xmm4 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[4] + i));
			xmm5 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[5] + i));
			xmm6 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[6] + i));
			xmm7 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[7] + i));

			xmm8  = _mm256_unpacklo_epi16(xmm0, xmm1);
			xmm9  = _mm256_unpackhi_epi16(xmm0, xmm1);
			xmm10 = _mm256_unpacklo_epi16(xmm2, xmm3);
			xmm11 = _mm256_unpackhi_epi16(xmm2, xmm3);
			xmm12 = _mm256_unpacklo_epi16(xmm4, xmm5);
			xmm13 = _mm256_unpackhi_epi16(xmm4, xmm5);
			xmm14 = _mm256_unpacklo_epi16(xmm6, xmm7);
			xmm15 = _mm256_unpackhi_epi16(xmm6, xmm7);

			xmm16 = _mm256_unpacklo_epi32(xmm8,  xmm10);
			xmm17 = _mm256_unpackhi_epi32(xmm8,  xmm10);
			xmm18 = _mm256_unpacklo_epi32(xmm12, xmm14);
			xmm19 = _mm256_unpackhi_epi32(xmm12, xmm14);
			xmm20 = _mm256_unpacklo_epi32(xmm9,  xmm11);
			xmm21 = _mm256_unpackhi_epi32(xmm9,  xmm11);
			xmm22 = _mm256_unpacklo_epi32(xmm13, xmm15);
			xmm23 = _mm256_unpackhi_epi32(xmm13, xmm15);

			xmm24 = _mm256_unpacklo_epi64(xmm16, xmm18);
			xmm25 = _mm256_unpackhi_epi64(xmm16, xmm18);
			xmm26 = _mm256_unpacklo_epi64(xmm17, xmm19);
			xmm27 = _mm256_unpackhi_epi64(xmm17, xmm19);
			xmm28 = _mm256_unpacklo_epi64(xmm20, xmm22);
			xmm29 = _mm256_unpackhi_epi64(xmm20, xmm22);
			xmm30 = _mm256_unpacklo_epi64(xmm21, xmm23);
			xmm31 = _mm256_unpackhi_epi64(xmm21, xmm23);

			_mm256_store_si256(dprofile + 4 * (i + 0) + j, xmm24);
			_mm256_store_si256(dprofile + 4 * (i + 1) + j, xmm25);
			_mm256_store_si256(dprofile + 4 * (i + 2) + j, xmm26);
			_mm256_store_si256(dprofile + 4 * (i + 3) + j, xmm27);
			_mm256_store_si256(dprofile + 4 * (i + 4) + j, xmm28);
			_mm256_store_si256(dprofile + 4 * (i + 5) + j, xmm29);
			_mm256_store_si256(dprofile + 4 * (i + 6) + j, xmm30);
			_mm256_store_si256(dprofile + 4 * (i + 7) + j, xmm31);

			xmm0 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[ 8] + i));
			xmm1 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[ 9] + i));
			xmm2 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[10] + i));
			xmm3 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[11] + i));
			xmm4 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[12] + i));
			xmm5 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[13] + i));
			xmm6 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[14] + i));
			xmm7 = _mm256_load_si256((__m256i *) (dup_score_matrix_word + d[15] + i));

			xmm8  = _mm256_unpacklo_epi16(xmm0, xmm1);
			xmm9  = _mm256_unpackhi_epi16(xmm0, xmm1);
			xmm10 = _mm256_unpacklo_epi16(xmm2, xmm3);
			xmm11 = _mm256_unpackhi_epi16(xmm2, xmm3);
			xmm12 = _mm256_unpacklo_epi16(xmm4, xmm5);
			xmm13 = _mm256_unpackhi_epi16(xmm4, xmm5);
			xmm14 = _mm256_unpacklo_epi16(xmm6, xmm7);
			xmm15 = _mm256_unpackhi_epi16(xmm6, xmm7);

			xmm16 = _mm256_unpacklo_epi32(xmm8,  xmm10);
			xmm17 = _mm256_unpackhi_epi32(xmm8,  xmm10);
			xmm18 = _mm256_unpacklo_epi32(xmm12, xmm14);
			xmm19 = _mm256_unpackhi_epi32(xmm12, xmm14);
			xmm20 = _mm256_unpacklo_epi32(xmm9,  xmm11);
			xmm21 = _mm256_unpackhi_epi32(xmm9,  xmm11);
			xmm22 = _mm256_unpacklo_epi32(xmm13, xmm15);
			xmm23 = _mm256_unpackhi_epi32(xmm13, xmm15);

			xmm24 = _mm256_unpacklo_epi64(xmm16, xmm18);
			xmm25 = _mm256_unpackhi_epi64(xmm16, xmm18);
			xmm26 = _mm256_unpacklo_epi64(xmm17, xmm19);
			xmm27 = _mm256_unpackhi_epi64(xmm17, xmm19);
			xmm28 = _mm256_unpacklo_epi64(xmm20, xmm22);
			xmm29 = _mm256_unpackhi_epi64(xmm20, xmm22);
			xmm30 = _mm256_unpacklo_epi64(xmm21, xmm23);
			xmm31 = _mm256_unpackhi_epi64(xmm21, xmm23);

			_mm256_store_si256(dprofile + 4 * (i +  8) + j, xmm24);
			_mm256_store_si256(dprofile + 4 * (i +  9) + j, xmm25);
			_mm256_store_si256(dprofile + 4 * (i + 10) + j, xmm26);
			_mm256_store_si256(dprofile + 4 * (i + 11) + j, xmm27);
			_mm256_store_si256(dprofile + 4 * (i + 12) + j, xmm28);
			_mm256_store_si256(dprofile + 4 * (i + 13) + j, xmm29);
			_mm256_store_si256(dprofile + 4 * (i + 14) + j, xmm30);
			_mm256_store_si256(dprofile + 4 * (i + 15) + j, xmm31);
		}
	}
#else
	//naive implementation
	short* pdst  = dprofile_word;
	
	for(int yidx = 0; yidx < 32; ++yidx){
	for(int j = 0; j < 4; ++j){
		for(int i = 0; i < 16; ++i){
			int xidx = dseq[j * 16 + i];	
				*(pdst++) = dup_score_matrix_word[yidx * 32 + xidx];

		}
	}}
	
#endif
}

void avxInitLocks(){
	//pthread_mutex_init(&idxMutex, NULL);
	//pthread_mutex_init(&cntMutex, NULL);

}

void avxDestroyLocks(){
	//pthread_mutex_destroy(&idxMutex);
	//pthread_mutex_destroy(&cntMutex);
} 

void avxCompute16(
		short* dup_score_matrix,
		char* dbSeq,
		int* vecPos,
		int numSeqs,
		int* result,
		int start_idx,
		char* sseQuery,
		size_t qlen
		)
{
	if(start_idx == -1)
		return;
	__m256i S, Q, R, T, M, Z, T0, T1;
	__m256i *hep, **qp;
	char * d_begin[N16_CHANNELS];
	__m256i dseqalloc[CDEPTH];

	//short hearray[qlen * 32];
	short* hearray = (short *)_mm_malloc(qlen * 2 * N16_CHANNELS * sizeof(short), 32);
	//memset(hearray, 0, sizeof(short) * 2 * qlen * N16_CHANNELS);
	
	char * dseq = (char *) &dseqalloc;
	char zero;

	short* dprofile_word = (short*)  _mm_malloc(4 * 16 * 32 * sizeof(short), 32);
	short**  qtable      = (short**) _mm_malloc(qlen * sizeof(short*), 32);

	memset(dprofile_word, 1, sizeof(short) * 4 * 16 * 32);

	for (size_t i = 0; i < qlen; ++i)
		qtable[i] = dprofile_word + 4 * 16 * sseQuery[i];	
	
	int seq_id[N16_CHANNELS];
	int next_id = 0;
	int done;

	Z = _mm256_set_epi16(
			0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,	0x8000,
			0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,	0x8000);

	T0 = _mm256_set_epi16(
			0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,0x0000,
			0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,0x8000);
	T1 = _mm256_set_epi16(
			0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,0x8000,
			0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,0x0000);
	
	short gap_open_penalty = 12;
	short gap_extend_penalty = 2;
	
	Q = _mm256_set1_epi16(gap_open_penalty);
	R = _mm256_set1_epi16(gap_extend_penalty);

	zero = 23;
	done = 0;
	S = Z;
	hep = (__m256i *)  hearray;
	qp  = (__m256i **) qtable;

	for(size_t qi = 0; qi < qlen; qi++){
			hep[2 * qi    ] = Z;
			hep[2 * qi + 1] = Z;
	}

	for(int c = 0; c < N16_CHANNELS; ++c){
		d_begin[c] = &zero;
		seq_id[c]  = -1;
	}

	int easy = 0;

	while(1){
		//printf("thread %d\n", omp_get_thread_num());
		if(easy){
			for (int c = 0; c < N16_CHANNELS; c++) {
				char v;
				for (int j = 0; j < CDEPTH; j++) {
					v = *(d_begin[c]);
					dseq[N16_CHANNELS * j + c] = v;
					if (v != 23)
						d_begin[c]++;
				}
				if (*(d_begin[c]) == 23)
					easy = 0;
			}

			dprofile_fill16(dprofile_word, dup_score_matrix, dseq);
			donormal16(&S, hep, qp, &Q, &R, qlen, &Z);
		}else{
			easy = 1;
			M = _mm256_setzero_si256();

			T = T0;//For lower 128-bit lane
			for (int c = 0; c < 16; c++) {
				if(c == 8)
					T = T1;//For upper 128-bit lane
				if (*(d_begin[c]) != 23) {
					for (int j = 0; j < CDEPTH; j++) {
						char v = *(d_begin[c]);
						dseq[N16_CHANNELS * j + c] = v;
						if (v!= 23)
							d_begin[c]++;
					}
					if (*(d_begin[c]) == 23)
						easy = 0;
				} else {
					M = _mm256_xor_si256(M, T);
					long cand_id = seq_id[c];
					if (cand_id >= 0) {
						/*save the alignment score*/
						int score = ((unsigned short*) &S)[c] ^ 0x8000;
						result[cand_id + start_idx] = score;
						done++;
					}

					if (next_id < numSeqs) {
						seq_id[c] = next_id;
						d_begin[c] = dbSeq + vecPos[next_id + start_idx];
						next_id++;
						for (int j = 0; j < CDEPTH; j++) {
							char v = *(d_begin[c]);
							//printf("%d ", v);
							dseq[N16_CHANNELS * j + c] = v;
							if (v != 23)
								d_begin[c]++;
						}
						if (*(d_begin[c]) == 23)
							easy = 0;
					} else {
						seq_id[c] = -1;
						d_begin[c] = &zero;
						for (int j = 0; j < CDEPTH; j++){
							dseq[N16_CHANNELS * j + c] = 23;
						}
					}
				}
				T = _mm256_slli_si256(T, 2);
			}

			if (done == numSeqs) {
				break;
			}
			dprofile_fill16(dprofile_word, dup_score_matrix, dseq);
			domasked16(&S, hep, qp, &Q, &R, qlen, &Z, &M);
		}
	}
	_mm_free(hearray);
	_mm_free(dprofile_word);
	_mm_free(qtable);
}
