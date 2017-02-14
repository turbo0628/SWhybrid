#include "CUDAKernel.h"

texture<int4, cudaTextureType2D, cudaReadModeElementType> queryProfileTexture;

#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

__device__ int globalCounter = -1;

__device__ inline void ONE_CELL_COMP_SCALAR(int& f,int& e,int& h,int& n, const int sub, const unsigned gapoe, const unsigned gape,int& S) {
	asm("add.sat.s32 %0, %1, %2;" : "=r"(h): "r"(h), "r"(sub));         //h = h + sub;
	asm("max.s32 %0, %1, %2;"     : "=r"(h): "r"(h), "r"(f));           //h = max(h, f);
	asm("max.s32 %0, %1, %2;"     : "=r"(h): "r"(h), "r"(e));           //h = max(h, e);
	asm("max.s32 %0, %1, %2;"     : "=r"(S): "r"(h), "r"(S));           //S = max(h, S);
	asm("mov.s32 %0, %1;"         : "=r"(n): "r"(h));                   //n = h;
	asm("sub.sat.s32 %0, %1, %2;" : "=r"(f): "r"(f), "r"(gape));        //f = f - gape;
	asm("sub.sat.s32 %0, %1, %2;" : "=r"(e): "r"(e), "r"(gape));        //e = e - gape;
	asm("sub.sat.s32 %0, %1, %2;" : "=r"(h): "r"(h), "r"(gapoe));       //h = h - gapoe;
	asm("max.s32 %0, %1, %2;"     : "=r"(f): "r"(h), "r"(f));           //f = max(h, f);
	asm("max.s32 %0, %1, %2;"     : "=r"(e): "r"(h), "r"(e));           //e = max(h, e);
}

__device__ inline void BLOCK_COMP_QP(int4* f, int4* h, int4* n, int& e, int& maxHH, int& bh, const int& sa, const int& sb, const int gapoe, const int gape){
	int4 sub;

	sub = tex2D(queryProfileTexture, sb, sa);
	ONE_CELL_COMP_SCALAR(f[0].x, e,     bh, n[0].x, sub.x, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[0].y, e, h[0].x, n[0].y, sub.y, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[0].w, e, h[0].y, n[0].w, sub.w, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[0].z, e, h[0].w, n[0].z, sub.z, gapoe, gape, maxHH);

	sub = tex2D(queryProfileTexture, sb + 1, sa);
	ONE_CELL_COMP_SCALAR(f[1].x, e, h[0].z, n[1].x, sub.x, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[1].y, e, h[1].x, n[1].y, sub.y, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[1].w, e, h[1].y, n[1].w, sub.w, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[1].z, e, h[1].w, n[1].z, sub.z, gapoe, gape, maxHH);


	sub = tex2D(queryProfileTexture, sb + 2, sa);
	ONE_CELL_COMP_SCALAR(f[2].x, e, h[1].z, n[2].x, sub.x, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[2].y, e, h[2].x, n[2].y, sub.y, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[2].w, e, h[2].y, n[2].w, sub.w, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[2].z, e, h[2].w, n[2].z, sub.z, gapoe, gape, maxHH);

	sub = tex2D(queryProfileTexture, sb + 3, sa);
	ONE_CELL_COMP_SCALAR(f[3].x, e, h[2].z, n[3].x, sub.x, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[3].y, e, h[3].x, n[3].y, sub.y, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[3].w, e, h[3].y, n[3].w, sub.w, gapoe, gape, maxHH);
	ONE_CELL_COMP_SCALAR(f[3].z, e, h[3].w, n[3].z, sub.z, gapoe, gape, maxHH);

	bh = h[3].z;
}
#define cudaGapOE 12
#define cudaGapExtend 2
#define QUERY_LENGTH_ALIGNED 16
#define DEPTH 4

__device__ int computeScalarQP(const int qlen,
			const unsigned* dbseq, 
			const unsigned  dblen, 
			short4 *global,
			size_t pitch
		) 
{
	//if(dblen >= 3072)
	//	return 0x0;
	int lane = threadIdx.x % 32; //thread index inside a warp

	int4 sa;
	int sb;

	/* int4 f[3], h[3], n[3], bh, be; */
	int4 f[4], h[4], n[4], bh;

	int4   init4= {0x80000000, 0x80000000, 0x80000000, 0x80000000};
	short4 szero4 = { 0, 0, 0, 0 };

	unsigned int gapoe = cudaGapOE; //copy into register
	unsigned int gape = cudaGapExtend;

	int dblenquad = dblen >> 2;

	//initialize
	for (int i = 0; i < dblenquad * 2; i++) {
		global[i * pitch] = szero4;
	}

	int maxHH = 0x80000000;
	int4     e = init4;

	for (int i = 0; i < qlen; i += QUERY_LENGTH_ALIGNED) {
		h[0] = n[0] = f[0] = init4;
		h[1] = n[1] = f[1] = init4;
		h[2] = n[2] = f[2] = init4;
		h[3] = n[3] = f[3] = init4;

		int iquad = i >> 2;
		int seqIdx = lane - warpSize;
		for (int jquad = 0; jquad != dblenquad; jquad++) {
			unsigned pac4 = dbseq[seqIdx += warpSize];
			sa.x = pac4 & 0xFF;
			pac4 >>= 8;
			sa.y = pac4 & 0xFF;
			pac4 >>= 8;
			sa.w = pac4 & 0xFF;
			pac4 >>= 8;
			sa.z = pac4 & 0xFF;

			sb = iquad;


			int gIdx = jquad * 2;
			short4 loadH = global[gIdx];
			short4 loadE = global[gIdx + 1];
			
			bh = make_int4(loadH.x | 0x80000000, 
					loadH.y | 0x80000000, 
					loadH.w | 0x80000000, 
					loadH.z | 0x80000000);
			e = make_int4(loadE.x | 0x80000000, 
					loadE.y | 0x80000000, 
					loadE.w | 0x80000000, 
					loadE.z | 0x80000000);

			//compute 4 columns in the matrix
			BLOCK_COMP_QP(f, h, n, e.x, maxHH, bh.x,  sa.x, sb, gapoe, gape);
			BLOCK_COMP_QP(f, n, h, e.y, maxHH, bh.y,  sa.y, sb, gapoe, gape);
			BLOCK_COMP_QP(f, h, n, e.w, maxHH, bh.w,  sa.w, sb, gapoe, gape);
			BLOCK_COMP_QP(f, n, h, e.z, maxHH, bh.z,  sa.z, sb, gapoe, gape);

			short4 saveH, saveE;
			saveH = make_short4(min(bh.x, 0x8000FFFF) & 0xFFFF, 
					min(bh.y, 0x8000FFFF) & 0xFFFF, 
					min(bh.w, 0x8000FFFF) & 0xFFFF, 
					min(bh.z, 0x8000FFFF) & 0xFFFF);

			/*max score will never overflow if H is not overflow
			  so we don't check E*/
			saveE = make_short4(e.x  & 0xFFFF, e.y  & 0xFFFF, e.w  & 0xFFFF, e.z  & 0xFFFF);
			global[gIdx    ] = saveH;
			global[gIdx + 1] = saveE;

		}
	}
	return maxHH ^ 0x80000000;
}
__global__ void Compute(
		unsigned* databaseSequence, 
		unsigned*	deviceMap,
	 	unsigned 	batchNum,
		unsigned 	queryLen,
		int 			*result,
		int4*    	globalArray,
		size_t   	globalPitch
		) 
{
	int lane = threadIdx.x % warpSize;
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = thread_id >> 5;
	int curWarpIdx = warp_id;

	short4 global[3072];

	/*atomic add should be deprecated*/
	if (lane == 0) {
		atomicAdd(&globalCounter, 1);
	}

	while (curWarpIdx < batchNum) {
		unsigned pos 	= deviceMap[curWarpIdx * 2];
		unsigned dblen 	= deviceMap[curWarpIdx * 2 + 1];
		unsigned *dbseq = databaseSequence + pos;

		int score = 0;
		//score = computeScalarQP(queryLen, dbseq, dblen, globalArray + thread_id, globalPitch);
		score = computeScalarQP(queryLen, dbseq, dblen, global, 1);
		result[curWarpIdx * warpSize + lane] = score;
		//result[curWarpIdx * warpSize + lane] = score - 0x80000000;
		//update shared index in a warp
		if (lane == 0) {
			curWarpIdx = atomicAdd(&globalCounter, 1);
		}
		//update local index of each thread
		curWarpIdx = __shfl(curWarpIdx, 0);
	}
	globalCounter = -1; //reset global variables for next call
}

void kernelLaunch(
		int 		numBlocks,
		int		numThreads,
		cudaStream_t& 	stream,
		unsigned*	deviceBuffer,
		unsigned*	deviceMap,
		unsigned	batchNum,
		unsigned	queryLen,
		int*		devResult,
		int4*		globalArray,
		size_t		globalPitch
		)
{
	Compute<<<numBlocks, numThreads, 0, stream>>>(
						deviceBuffer,
						deviceMap,
						batchNum, 
						queryLen,
						devResult,
						globalArray,
						globalPitch
						);
	
}


void bindQueryPrf(cudaArray_t cu_array){
	queryProfileTexture.addressMode[0] = cudaAddressModeClamp;
	queryProfileTexture.addressMode[1] = cudaAddressModeClamp;
	queryProfileTexture.filterMode = cudaFilterModePoint;
	queryProfileTexture.normalized = false;
	cudaBindTextureToArray(queryProfileTexture, cu_array, queryProfileTexture.channelDesc);
}


