#include "AVX512Kernel.h"
#include <immintrin.h>
#include <omp.h>
#include <pthread.h>

#define CDEPTH 4


 void pfalloc(char* query, int qlen, int threadNum, int* g_profile, int** g_qtable){
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
}

 void pffree(){
}

 void swCalcScoreProfile(
		int * dprofile, 
		int * score_matrix,
		char * dbseq2
		)
{
	__m512i vone, score, score2; 
	__m128i bdbseq1, bdbseq2, bdbseq3, bdbseq4;
	__m512i vdbseq1, vdbseq2, vdbseq3, vdbseq4;
	__m512i result, result2, result3, result4;
	__mmask16 mask, maskf, mask2, maskf2, mask3, maskf3, mask4, maskf4;
	__m512i vhex = _mm512_set1_epi32(16);
	__m128i *dbseq = (__m128i*) dbseq2;
	/*load and convert int8 to int32*/
	bdbseq1 = _mm_load_si128(dbseq); 
	bdbseq2 = _mm_load_si128(dbseq + 1); 
	bdbseq3 = _mm_load_si128(dbseq + 2); 
	bdbseq4 = _mm_load_si128(dbseq + 3); 
	vdbseq1 = _mm512_cvtepi8_epi32(bdbseq1);
	vdbseq2 = _mm512_cvtepi8_epi32(bdbseq2);
	vdbseq3 = _mm512_cvtepi8_epi32(bdbseq3);
	vdbseq4 = _mm512_cvtepi8_epi32(bdbseq4);
	mask = _mm512_cmp_epi32_mask(vdbseq1, vhex, _MM_CMPINT_LT);//check if smaller or larger than 16
	maskf = _mm512_knot(mask);
	mask2 = _mm512_cmp_epi32_mask(vdbseq2, vhex, _MM_CMPINT_LT);
	maskf2 = _mm512_knot(mask2);

	mask3 = _mm512_cmp_epi32_mask(vdbseq3, vhex, _MM_CMPINT_LT);
	maskf3 = _mm512_knot(mask3);

	mask4 = _mm512_cmp_epi32_mask(vdbseq4, vhex, _MM_CMPINT_LT);
	maskf4 = _mm512_knot(mask4);

	for (int i = 0; i < 24; i++)
	{
		score = _mm512_load_epi32(score_matrix + (i << 5));
		score2 = _mm512_load_epi32(score_matrix + (i << 5) + 16);
		result = _mm512_mask_permutevar_epi32(score, mask, vdbseq1, score);
		vdbseq1 = _mm512_sub_epi32(vdbseq1, vhex);
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
}

 inline void ONE_CELL_UPDATE(
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
	vN = vH;
	vH = _mm512_sub_epi32(vH, Q);//H = H - Q
	vE = _mm512_max_epi32(vH, vE);// E = max(H, E)
	vF = _mm512_max_epi32(vH, vF);//F = ax(H, F)
}

 void swSearchTileSinglePass(
		int ** q_start,
		int *savedH,
		int * savedE,
		__m512i R,
		__m512i Q,
		int qlen,
		int *S
		)
{
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
}

/*this version differs in memory access*/
 void swSearchTileMultiPass(int** q_start, int startPos, int *savedH, int * savedE, int* backH, int* backF,__m512i R, __m512i Q, int qlen, int *S) {
	register __m512i vZero, vE, vP, vS, vHload;
	register __m512i vH[4], vN[4], vF[4];
	int i, j;
	vZero = _mm512_setzero_epi32();
	vE = _mm512_load_epi32(savedE);
	vS = _mm512_load_epi32(S);
	q_start += startPos;

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
}

 void avx512Compute(char* query, 
		int qlen, 
		int gapoe,
		int gape,
		char* deviceDBSeq, 
		int* deviceMap,
	 	size_t batchNum, 
		int* devResult, 
		int* deviceMatrix, 
		int& globalCounter, 
		int* g_profile, 
		int** g_qtable
		)
{
	int tid = omp_get_thread_num();
	int* dprofile = g_profile + 4 * 16 * 32 * tid;
	int** qtable  = g_qtable  + qlen * tid;
	const int restrictedQueryLen = 2048;

	int localIdx = -1;

#pragma omp atomic capture
	localIdx = ++globalCounter;

	 __m512i Q, R;
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

#pragma omp atomic capture
			localIdx = ++globalCounter;
		}

	} else {
		const int threshold = 750;
		int pass, bodylen, lastlen;
		pass = (qlen + threshold - 1) 	/  threshold;
		bodylen = (qlen + pass - 1) / pass;
		lastlen = qlen - bodylen * (pass - 1);
		//printf("pass %d, bodylen %d, lastlen %d\n", pass, bodylen, lastlen);	
		
		int *savedE = (int*) _mm_malloc(64*(bodylen+1),64);
		int *savedH = (int*) _mm_malloc(64*(bodylen+1),64);
		int *backH  = (int*) _mm_malloc(64*3072,64);
		int *backF  = (int*) _mm_malloc(64*3072,64);

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
				memset(savedE, 0, 16 * (bodylen +1) * sizeof(int));
				memset(savedH, 0, 16 * (bodylen +1) * sizeof(int));
				if(j == pass - 1)
					curQueryLen = lastlen;	
				else
					curQueryLen = bodylen;
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
					
				//printf("S0 %d\n", S[0]);
			}
#pragma omp atomic capture
			localIdx = ++globalCounter;
		}
		_mm_free(savedE);
		_mm_free(savedH);
		_mm_free(backH);
		_mm_free(backF);
	}
}
