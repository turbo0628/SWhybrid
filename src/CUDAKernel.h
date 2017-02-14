#ifndef _CUDA_KERNEL_H
#define _CUDA_KERNEL_H

#include <cuda_runtime.h>

#include "Worker.h"//for DB map


void kernelLaunch(
		int 			numBlocks,
		int			numThreads,
		cudaStream_t& 		stream,
		unsigned*		deviceBuffer,
		unsigned*		deviceMap,
		unsigned		batchNum,
		unsigned		queryLen,
		int*			devResult,
		int4*			globalArray,
		size_t			globalPitch
		);

void bindQueryPrf(cudaArray_t cudaPtxQueryPrf);

#endif
