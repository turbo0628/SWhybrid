#include "KNCKernel.h"
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>

#define BLOCK_LENGTH 750
#define CDEPTH 4

const int threshold = 750;

__ONMIC__ static const int gape = 2;
__ONMIC__ static const int gapoe = 12;
__ONMIC__ const int restrictedQueryLen = 900;


__ONMIC__ void pfalloc(char* query, int qlen, int threadNum, int* g_profile, int** g_qtable){
#ifdef __MIC__
	//for locks
	//pthread_mutex_init(&idxMutex, NULL);
	//for score profile
	//g_profile = (int*) _mm_malloc(4 * 16 * 32 * threadNum * sizeof(int), 64);
	//g_qtable = (int**) _mm_malloc(qlen * sizeof(int*) * threadNum, 64);
	memset(g_profile, 0, sizeof(int) * 4 * 16 * 32 * threadNum);

	//construct the 2-dimensional qtable
	int** qtable = g_qtable;
	int*  profile = g_profile;
	for(int j = 0; j != threadNum; ++j){
		for (int i = 0; i < qlen; ++i){
			qtable[i] = profile + 64 * query[i];
		}
		profile += 4 * 16 * 32;
		qtable  += qlen;
	}
	//printf("g_profile %x\n", g_profile);
	//printf("g_qtable %x\n", g_qtable);
#endif
}

__ONMIC__ void pffree(){
#ifdef __MIC__
	//pthread_mutex_destroy(&idxMutex);
	//printf("g_profile %x\n", g_profile);
	//printf("g_qtable %x\n", g_qtable);
	//_mm_free(g_profile);
	//_mm_free(g_qtable);
#endif
}

__ONMIC__ void swCalcScoreProfile(
		int * dprofile, 
		int * score_matrix,
		char * dbseq2
		)
{
#ifdef __MIC__
	__m512i vone, score, score2, result, vdbseq1, vdbseq2, vdbseq3, vdbseq4,
	result2, result3, result4;
	__mmask16 mask, maskf, mask2, maskf2, mask3, maskf3, mask4, maskf4;
	__m512i vhex = _mm512_set1_epi32(16);

	/*ext load to extend int8 to nit32*/
	vdbseq1 = _mm512_extload_epi32(dbseq2, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0); 
	mask = _mm512_cmp_epi32_mask(vdbseq1, vhex, _MM_CMPINT_LT);//check if smaller or larger than 16
	maskf = _mm512_knot(mask);

	vdbseq2 = _mm512_extload_epi32(dbseq2 + 16, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0); 
	mask2 = _mm512_cmp_epi32_mask(vdbseq2, vhex, _MM_CMPINT_LT);
	maskf2 = _mm512_knot(mask2);

	vdbseq3 = _mm512_extload_epi32(dbseq2 + 32, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0); 
	mask3 = _mm512_cmp_epi32_mask(vdbseq3, vhex, _MM_CMPINT_LT);
	maskf3 = _mm512_knot(mask3);

	vdbseq4 = _mm512_extload_epi32(dbseq2 + 48, _MM_UPCONV_EPI32_SINT8, _MM_BROADCAST32_NONE, 0); 
	mask4 = _mm512_cmp_epi32_mask(vdbseq4, vhex, _MM_CMPINT_LT);
	maskf4 = _mm512_knot(mask4);

	for (int i = 0; i < 24; i++)
	{
		score = _mm512_load_epi32(score_matrix + (i << 5));
		score2 = _mm512_load_epi32(score_matrix + (i << 5) + 16);
		//get the scores held in the first score register
		result = _mm512_mask_permutevar_epi32(score, mask, vdbseq1, score);
		//subtract 16 of each index to get the index in the second score register
		vdbseq1 = _mm512_sub_epi32(vdbseq1, vhex);
		//get the scores held in the second score register
		result = _mm512_mask_permutevar_epi32(result, maskf, vdbseq1, score2);
		_mm512_store_epi32(dprofile + 64 * i, result);

		result2 = _mm512_mask_permutevar_epi32(score, mask2, vdbseq2, score);
		vdbseq2 = _mm512_sub_epi32(vdbseq2, vhex);
		result2 = _mm512_mask_permutevar_epi32(result2, maskf2, vdbseq2, score2);
		_mm512_store_epi32(dprofile + 64 * i + 16, result2);

		result3 = _mm512_mask_permutevar_epi32(score, mask3, vdbseq3, score);
		vdbseq3 = _mm512_sub_epi32(vdbseq3, vhex);
		result3 = _mm512_mask_permutevar_epi32(result3, maskf3, vdbseq3, score2);
		_mm512_store_epi32(dprofile + 64 * i + 32, result3);

		result4 = _mm512_mask_permutevar_epi32(score, mask4, vdbseq4, score);
		vdbseq4 = _mm512_sub_epi32(vdbseq4, vhex);
		result4 = _mm512_mask_permutevar_epi32(result4, maskf4, vdbseq4, score2);
		_mm512_store_epi32(dprofile + 64 * i + 48, result4);
	}
#endif
}

__ONMIC__ inline void ONE_CELL_UPDATE(
		__m512i& vH,
		__m512i& vN,
		__m512i& vE,
		__m512i& vF,
		__m512i& vS,
		__m512i& vP,
		__m512i& vZero,
		__m512i& R,
		__m512i& Q
		)
{
	
			vH = _mm512_add_epi32(vH, vP);//H = H + p[q]
			vH = _mm512_max_epi32(vH, vF);//H = max(H, F)
			vH = _mm512_max_epi32(vH, vE);//H = max(H, E)
			vH = _mm512_max_epi32(vH, vZero);//H = max(H,0)
			vS = _mm512_max_epi32(vS, vH);//S = max(S, H)
			vF = _mm512_sub_epi32(vF, R);//F = F - R
			vE = _mm512_sub_epi32(vE, R);//E = E - R
			vN = _mm512_mask_mov_epi32(vN, 0xffff, vH);// N = H
			vH = _mm512_sub_epi32(vH, Q);//H = H - Q
			vE = _mm512_max_epi32(vH, vE);// E = max(H, E)
			vF = _mm512_max_epi32(vH, vF);//F = ax(H, F)
}

__ONMIC__ void swSearchTileSinglePass(
		int ** q_start,
		int *savedH,
		int * savedE,
		__m512i R,
		__m512i Q,
		int qlen,
		int *S
		)
{
#ifdef __MIC__
	register __m512i vZero, vE, vP, vS, vHload;
	register __m512i vH[4], vN[4], vF[4];
	int i, j;

	vZero = _mm512_setzero_epi32();
	vE = _mm512_load_epi32(savedE);
	vS = _mm512_load_epi32(S);

	/*
	 * Calculation for the first row in the tile
	 */
	for (i = 0; i < 4; i++)
	{
		vH[i] = _mm512_setzero_epi32();
		vF[i] = _mm512_setzero_epi32();
		//loading sub score
		vP = _mm512_load_epi32((q_start[0] + 16 * i));
		ONE_CELL_UPDATE(vH[i], vN[i], vE, vF[i], vS, vP, vZero, R, Q);
	}
	_mm512_store_epi32(savedE, vE);
	//Calculate along the query.
	for (j = 1; j < qlen; j++)
	{
		//1st cell in the row
		//load H and then update
		vHload = _mm512_load_epi32(savedH + (j - 1) * 16);
		_mm512_store_epi32(savedH + (j - 1) * 16, vN[3]);
		vE = _mm512_load_epi32(savedE + 16 * j);
		vP = _mm512_load_epi32(q_start[j]);
		ONE_CELL_UPDATE(vHload, vH[0], vE, vF[0], vS, vP, vZero, R, Q);
		//update remaining 3 cells in the row
		for (i = 1; i < 4; i++)
		{
			vP = _mm512_load_epi32(q_start[j] + 16 * i);
			ONE_CELL_UPDATE(vN[i-1], vH[i], vE, vF[i], vS, vP, vZero, R, Q);
		}
		_mm512_store_epi32(savedE + j * 16, vE);
		++j;
		if (j == qlen)
			break;
		//swap vH and vN
		vHload = _mm512_load_epi32(savedH + (j - 1) * 16);
		_mm512_store_epi32(savedH + (j - 1) * 16, vH[3]);
		vE = _mm512_load_epi32(savedE + 16 * j);
		vP = _mm512_load_epi32(q_start[j]);
		ONE_CELL_UPDATE(vHload, vN[0], vE, vF[0], vS, vP, vZero, R, Q);
		for (i = 1; i < 4; i++)
		{
			vP = _mm512_load_epi32(q_start[j] + 16 * i);
			ONE_CELL_UPDATE(vH[i-1], vN[i], vE, vF[i], vS, vP, vZero, R, Q);
		}
		_mm512_store_epi32(savedE + j * 16, vE);
	}
	_mm512_store_epi32(S, vS);
#endif
}

/*this version differs in memory access*/
__ONMIC__ void swSearchTileMultiPass(int** q_start, int startPos, int *savedH, int * savedE, int* backH, int* backF,__m512i R, __m512i Q, int qlen, int *S) {
#ifdef __MIC__
	register __m512i vZero, vE, vP, vS, vHload;
	register __m512i vH[4], vN[4], vF[4];
	int i, j;
	vZero = _mm512_setzero_epi32();
	vE = _mm512_load_epi32(savedE);
	vS = _mm512_load_epi32(S);
	q_start += startPos > 0 ? startPos : 0;

	/*
	 * Calculation for the first row in the tile
	 * If this is the starting pass, we don't have to load the context from backH and backF
	 */
	if(startPos == 0){
		for (i = 0; i < 4; i++)
		{
			vH[i] = _mm512_setzero_epi32();
			vF[i] = _mm512_setzero_epi32();
			vP = _mm512_load_epi32((q_start[0] + 16 * i));
			ONE_CELL_UPDATE(vH[i], vN[i], vE, vF[i], vS, vP, vZero, R, Q);
		}
	}else{
		for(i = 0; i != 4; ++i){
			vH[i] = _mm512_load_epi32(backH + i * 16);
			vF[i] = _mm512_load_epi32(backF + i * 16);
		}
		vHload = _mm512_load_epi32(savedH - 16);
		_mm512_store_epi32(savedH - 16, vH[3]);
		vP = _mm512_load_epi32(q_start[0]);

		ONE_CELL_UPDATE(vHload, vN[0], vE, vF[0], vS, vP, vZero, R, Q);
		//update remaining 3 cells in the row
		for (i = 1; i < 4; i++)
		{
			vP = _mm512_load_epi32(q_start[0] + 16 * i);
			ONE_CELL_UPDATE(vH[i-1], vN[i], vE, vF[i], vS, vP, vZero, R, Q);
		}
	}//startPos

	_mm512_store_epi32(savedE, vE);

	/*
	 * Calculate one tile along the query
	 * Each tile occupies 4 columns
	 * Row index begins from 1 for we have already finshed computing in the fist row
	 * When computing in a tile of a single pass has finshed, the context is stored in backH and backF
	 */
	for (j = 1; j < qlen; j++)
	{
		vHload = _mm512_load_epi32(savedH + (j - 1) * 16);
		_mm512_store_epi32(savedH + (j - 1) * 16, vN[3]);
		vE = _mm512_load_epi32(savedE + 16 * j);

		vP = _mm512_load_epi32(q_start[j]);
		ONE_CELL_UPDATE(vHload, vH[0], vE, vF[0], vS, vP, vZero, R, Q);

		/*Store context*/
		if(j == qlen - 1 ){
			_mm512_store_epi32(backH, vH[0]);
			_mm512_store_epi32(backF, vF[0]);
		}

		for (i = 1; i < 4; i++)
		{
			vP = _mm512_load_epi32(q_start[j] + 16 * i);
			ONE_CELL_UPDATE(vN[i-1], vH[i], vE, vF[i], vS, vP, vZero, R, Q);

			/*Store Context for next pass*/
			if(j == qlen - 1){
				_mm512_store_epi32(backH + i * 16, vH[i]);
				_mm512_store_epi32(backF + i * 16, vF[i]);
			}	
		}

		/*Save E*/
		_mm512_store_epi32(savedE + j * 16, vE);

		++j;

		if (j == qlen)
			break;
		
		vHload = _mm512_load_epi32(savedH + (j - 1) * 16);
		/*Save H*/
		_mm512_store_epi32(savedH + (j - 1) * 16, vH[3]);
		vE = _mm512_load_epi32(savedE + 16 * j);
		vP = _mm512_load_epi32(q_start[j]);
		ONE_CELL_UPDATE(vHload, vN[0], vE, vF[0], vS, vP, vZero, R, Q);

		/*save conatext for next pass*/
		if(  j == qlen - 1){
			_mm512_store_epi32(backF, vF[0]);
			_mm512_store_epi32(backH, vN[0]); 
		}

		for (i = 1; i < 4; i++){
			vP = _mm512_load_epi32(q_start[j] + 16 * i);
			ONE_CELL_UPDATE(vH[i-1], vN[i], vE, vF[i], vS, vP, vZero, R, Q);
			if(j == qlen - 1){
				_mm512_store_epi32(backH + i * 16, vN[i]);
				_mm512_store_epi32(backF + i * 16, vF[i]);
			}	
		}
		_mm512_store_epi32(savedE + j * 16, vE);
	}
	_mm512_store_epi32(S, vS);
#endif
}

__ONMIC__ void Compute(char* query, 
		int   qlen, 
		char* deviceDBSeq, 
		int* deviceMap,
	 	size_t batchNum, 
		int* devResult, 
		int* deviceMatrix, 
		int& globalCounter, 
		int pass,
		int bodyQueryLen,
		int lastQueryLen,
		int* g_profile, 
		int** g_qtable
		)
{
#ifdef __MIC__
	int tid = omp_get_thread_num();
	int* dprofile = g_profile + 4 * 16 * 32 * tid;
	int** qtable  = g_qtable  + qlen * tid;

	int localIdx = -1;

	//pthread_mutex_lock(&idxMutex);
#pragma omp atomic capture
	localIdx = ++globalCounter;
	//pthread_mutex_unlock(&idxMutex);
	__ONMIC__ __m512i Q, R;
	Q = _mm512_set1_epi32(12);
	R = _mm512_set1_epi32(2);
	int *S = NULL;

	if(qlen < restrictedQueryLen){
		__declspec(align(64)) int savedE[16 * qlen];
		__declspec(align(64)) int savedH[16 * qlen];
		while (localIdx < batchNum && localIdx > -1)
		{
			S = devResult + localIdx * 16;
			int pos = deviceMap[localIdx * 2];
			int dblen = deviceMap[localIdx * 2 + 1];
			char* dbseq = deviceDBSeq + pos;
			memset(savedE, 0, 16 * qlen * sizeof(int));
			memset(savedH, 0, 16 * qlen * sizeof(int));
			memset(S, 0, 16 * sizeof(int));
			for (int i = 0; i < dblen >> 2; ++i)
			{
				swCalcScoreProfile(dprofile, deviceMatrix, dbseq + i * 64);
				swSearchTileSinglePass(qtable, savedH, savedE, R, Q, qlen, S);
			}

			//pthread_mutex_lock(&idxMutex);
#pragma omp atomic capture
			localIdx = ++globalCounter;
			//pthread_mutex_unlock(&idxMutex);
		}

	} else {
		__declspec(align(64)) int savedE[16 * BLOCK_LENGTH + 16];
		__declspec(align(64)) int savedH[16 * BLOCK_LENGTH + 16];
		__declspec(align(64)) int backH[16 * 3072];
		__declspec(align(64)) int backF[16 * 3072];
		while (localIdx < batchNum && localIdx > -1)
		{
			
			S = devResult + localIdx * 16;
			int pos = deviceMap[localIdx * 2];
			int dblen = deviceMap[localIdx * 2 + 1];
			char* dbseq = deviceDBSeq + pos;
			memset(S, 0, 16 * sizeof(int));
			int startPos = 0;
			int curQueryLen;
			for(int j = 0; j < pass; ++j){
				memset(savedE, 0, 16 * (bodyQueryLen +1) * sizeof(int));
				memset(savedH, 0, 16 * (bodyQueryLen +1) * sizeof(int));
				if(j == pass - 1)
					curQueryLen = lastQueryLen;	
				else
					curQueryLen = bodyQueryLen;
				for (int i = 0; i != dblen >> 2; ++i)
				{
					swCalcScoreProfile(
							dprofile, 
							deviceMatrix, 
							dbseq + i * 64
							);
					swSearchTileMultiPass(
							qtable, 
							startPos, 
							savedH + 16, 
							savedE + 16, 
							backH  + i * 64, 
							backF + i * 64,
							R, Q, 
							curQueryLen, 
							S
							);
				}
				startPos += curQueryLen;
			}
			//pthread_mutex_lock(&idxMutex);
#pragma omp atomic capture
			localIdx = ++globalCounter;
			//pthread_mutex_unlock(&idxMutex);
		}
	}
#endif
}
