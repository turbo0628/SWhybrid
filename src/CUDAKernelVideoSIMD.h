#ifndef _CUDA_KERNEL_VIDEOSIMD_H
#define _CUDA_KERNEL_VIDEOSIMD_H

#include <cuda_runtime.h>

#include "Worker.h"//for DB map


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
		);

void bindQueryPrfVariant(cudaArray_t cudaPtxQueryPrf);

#endif
