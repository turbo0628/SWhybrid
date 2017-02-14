#include "CUDAKernelVideoSIMD.h"


__device__ __constant__ int cudaQueryAlignedLen;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }


__device__ int globalCounter = -1;

/*query profiles*/
texture<short4, 3, cudaReadModeElementType> InterQueryPrfPtx;

__device__ inline void ONE_CELL_COMP_QUAD(int& f,int& e, int& h, int& n, const int sub, const unsigned gapoe, const unsigned gape,int& S) {
	asm("vadd4.s32.s32.s32.sat %0, %1, %2, %3;" : "=r"(h) : "r"(h), "r"(sub),   "r"(0));	
	asm("vmax4.s32.s32.s32 %0, %1, %2, %3;"     : "=r"(h) : "r"(h), "r"(f),      "r"(0));	
	asm("vmax4.s32.s32.s32 %0, %1, %2, %3;"     : "=r"(h) : "r"(h), "r"(e),      "r"(0));	
	asm("vmax4.s32.s32.s32 %0, %1, %2, %3;"     : "=r"(S) : "r"(h), "r"(S),      "r"(0));	
	asm("mov.s32 %0, %1;" : "=r"(n) : "r"(h));
	asm("vsub4.s32.s32.s32.sat %0, %1, %2, %3;" : "=r"(f) : "r"(f),"r"(gape),    "r"(0)); 
	asm("vsub4.s32.s32.s32.sat %0, %1, %2, %3;" : "=r"(e) : "r"(e), "r"(gape), "r"(0));	
	asm("vsub4.s32.s32.s32.sat %0, %1, %2, %3;" : "=r"(h) : "r"(h), "r"(gapoe),  "r"(0));	
	asm("vmax4.s32.s32.s32 %0, %1, %2, %3;"     : "=r"(e) : "r"(h), "r"(e),    "r"(0));	
	asm("vmax4.s32.s32.s32 %0, %1, %2, %3;"     : "=r"(h) : "r"(h), "r"(f),      "r"(0));	
}
__device__ inline void BLOCK_COMP_QP_QUAD(int4* f, int4* h, int4* p, int& e, int& maxHH, int& bh, const int4& sa, const int2& sb, const int gapoe, const int gape){
	/*
	 *	 lanes x|y|z|w
	 *         1|2|3|4
	 */

	//loading substitution scores from query profile
	//h[0].x = bh;
	short4 qprf  = tex3D(InterQueryPrfPtx, sa.x, sa.y, sb.x);
	short4 qprf2 = tex3D(InterQueryPrfPtx, sa.z, sa.w, sb.x);

	int4 sub = make_int4(
			(qprf.x << 16) | (qprf2.x & 0x0ffff),
			(qprf.y << 16) | (qprf2.y & 0x0ffff),
			(qprf.z << 16) | (qprf2.z & 0x0ffff),
			(qprf.w << 16) | (qprf2.w & 0x0ffff));

	ONE_CELL_COMP_QUAD(f[0].x, e,      bh, p[0].x, sub.x, gapoe, gape, maxHH);
	ONE_CELL_COMP_QUAD(f[0].y, e,  h[0].x, p[0].y, sub.y, gapoe, gape, maxHH);
	ONE_CELL_COMP_QUAD(f[0].w, e,  h[0].y, p[0].w, sub.w, gapoe, gape, maxHH);
	ONE_CELL_COMP_QUAD(f[0].z, e,  h[0].w, p[0].z, sub.z, gapoe, gape, maxHH);


	//loading substitution scores from query profile
	qprf = tex3D(InterQueryPrfPtx, sa.x, sa.y, sb.y);
	qprf2 = tex3D(InterQueryPrfPtx, sa.z, sa.w, sb.y);
	sub = make_int4((qprf.x << 16) | (qprf2.x & 0x0ffff),
			(qprf.y << 16) | (qprf2.y & 0x0ffff),
			(qprf.z << 16) | (qprf2.z & 0x0ffff),
			(qprf.w << 16) | (qprf2.w & 0x0ffff));

	ONE_CELL_COMP_QUAD(f[1].x, e, h[0].z, p[1].x, sub.x, gapoe, gape, maxHH);
	ONE_CELL_COMP_QUAD(f[1].y, e, h[1].x, p[1].y, sub.y, gapoe,gape, maxHH);
	ONE_CELL_COMP_QUAD(f[1].w, e, h[1].y, p[1].w, sub.w, gapoe,gape, maxHH);
	ONE_CELL_COMP_QUAD(f[1].z, e, h[1].w, p[1].z, sub.z, gapoe,gape, maxHH);

	bh = h[1].z;
}

//#define MAX_SEQ_LENGTH_THRESHOLD 3072
#define QUERY_SEQ_LENGTH_ALIGNED 8

__device__ void computeQuad(const int qlen, const uint4* dbseq, const int dblen, int& maxHH, int4* externGlobal) {
/*
	if(dblen >= MAX_SEQ_LENGTH_THRESHOLD){
		maxHH = 0x7F7F7F7F;
		return;
	}
*/	
	int i, j;
	//int lane_id = threadIdx.x & 31;
	int4 sa;
	int2 sb;
	int4 h[2], p[2], f[2];
	int4 bh, be;
	int GAPOE = 12;
	int GAPE = 2;
	int gapoe = (GAPOE << 24) | (GAPOE << 16) | (GAPOE << 8) | GAPOE;
	int gape = (GAPE << 24) | (GAPE << 16) | (GAPE << 8) | GAPE;
	int4 zero = make_int4(0x80808080, 0x80808080, 0x80808080, 0x80808080);
	int2 zero2 = make_int2(0x80808080, 0x80808080);

	maxHH = 0x80808080;
	for(i = 0; i <= dblen / 2; ++i){
		externGlobal[i << 5] = zero;//interleaved
	}
	for (i = 1; i <= qlen; i += QUERY_SEQ_LENGTH_ALIGNED) {

		h[0] = p[0] = f[0] = zero;
		h[1] = p[1] = f[1] = zero;

		/*get the index for query profile*/
		sb.x = i >> 2;
		sb.y = sb.x + 1;

		int laneId = threadIdx.x & 31;
		int seqIdx = laneId - 32;
		for (j = 0; j < dblen; j += 4) 
		{
			//load the packed 4 residues from 4 sequences
			uint4 pack = dbseq[seqIdx += 32];

			int jquad = j >> 2;
			int gidx = jquad * 2;
			bh = externGlobal[gidx << 5];
			be = externGlobal[(gidx + 1)<< 5];
			//compute the cell block SEQ_LENGTH_ALIGNED x 4 

			sa = make_int4(pack.x & 0x0FF, pack.y & 0x0FF, pack.z & 0x0FF, pack.w & 0x0FF);
			pack.x >>= 8;
			pack.y >>= 8;
			pack.z >>= 8;
			pack.w >>= 8;

			BLOCK_COMP_QP_QUAD(f, h, p, be.x, maxHH, bh.x, sa, sb, gapoe, gape);	
			sa = make_int4(pack.x & 0x0FF, pack.y & 0x0FF, pack.z & 0x0FF, pack.w & 0x0FF);
			pack.x >>= 8;
			pack.y >>= 8;
			pack.z >>= 8;
			pack.w >>= 8;

			BLOCK_COMP_QP_QUAD(f, p, h, be.y, maxHH, bh.y, sa, sb, gapoe, gape);	
			sa = make_int4(pack.x & 0x0FF, pack.y & 0x0FF, pack.z & 0x0FF, pack.w & 0x0FF);
			pack.x >>= 8;
			pack.y >>= 8;
			pack.z >>= 8;
			pack.w >>= 8;
			BLOCK_COMP_QP_QUAD(f, h, p, be.w, maxHH, bh.w, sa, sb, gapoe, gape);	
			sa = make_int4(pack.x & 0x0FF, pack.y & 0x0FF, pack.z & 0x0FF, pack.w & 0x0FF);
			BLOCK_COMP_QP_QUAD(f, p, h, be.z, maxHH, bh.z, sa, sb, gapoe, gape);	

			externGlobal[gidx << 5] = bh;
			externGlobal[(gidx+1)  << 5] = be;
		}
	}
	maxHH ^= 0x80808080;
}	


__global__ void Compute(
		uint4* databaseSequence, 
		unsigned* deviceMap,
	 	unsigned batchNum,
		unsigned queryLen/*, int4* bh, int4* be*/,
		int *result,
		int4* global
		) 
{
	int lane       = threadIdx.x & 31;
	int thread_id  = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id    = thread_id >> 5;
	int curWarpIdx = warp_id;
	int score;

	if(lane == 0){
		atomicAdd(&globalCounter, 1);
	}

	while (curWarpIdx < batchNum) {

		unsigned pos   = deviceMap[curWarpIdx * 2];
		unsigned dblen = deviceMap[curWarpIdx * 2 + 1];

		uint4*dbseq = databaseSequence + pos;

		int resultIdx = (curWarpIdx * 32 + lane) << 2;

		computeQuad(queryLen, dbseq, dblen, score, global + pos*2 + lane);

		/*save results*/
		result[resultIdx + 3] =  score 	      & 0x0FF ;
		result[resultIdx + 2] = (score >> 8)  & 0x0FF ;
		result[resultIdx + 1] = (score >> 16) & 0x0FF ;
		result[resultIdx    ] = (score >> 24) & 0x0FF ;

	
		//update shared index in a warp
		if (lane == 0) {
			curWarpIdx = atomicAdd(&globalCounter, 1);
		}
		//update local index of each thread
		curWarpIdx = __shfl(curWarpIdx, 0);
	}
	globalCounter = -1;
}

void kernelLaunch(
		int 		numBlocks,
		int		numThreads,
		cudaStream_t& 	stream,
		uint4*		deviceBuffer,
		unsigned*	deviceMap,
		unsigned	batchNum,
		unsigned	queryLen,
		int*		devResult,
		int4*		globalArray
		)
{
	Compute<<<numBlocks, numThreads, 0, stream>>>(
						deviceBuffer,
						deviceMap,
						batchNum, 
						queryLen,
						devResult,
						globalArray
						);
	
}

void bindQueryPrfVariant(cudaArray_t cudaPtxQueryPrf){
	InterQueryPrfPtx.filterMode = cudaFilterModePoint;
	InterQueryPrfPtx.normalized = false;
	InterQueryPrfPtx.addressMode[0] = cudaAddressModeClamp;
	InterQueryPrfPtx.addressMode[1] = cudaAddressModeClamp;
	InterQueryPrfPtx.addressMode[2] = cudaAddressModeClamp;

	cudaChannelFormatDesc short4_channelDesc =
		cudaCreateChannelDesc<short4>();
	CUDA_CHECK_RETURN(cudaBindTextureToArray(InterQueryPrfPtx, cudaPtxQueryPrf, short4_channelDesc));
}
