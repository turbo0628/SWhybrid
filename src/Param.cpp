#include "Param.h"

#include <unistd.h>//for cpu core number
#ifdef WITH_KNC
#include <offload.h>
#endif

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

void Param::sysConf(){
	cpuAvailable = (int) sysconf(_SC_NPROCESSORS_ONLN);
	printf("number of available CPUs: %d\n", cpuAvailable);
#ifdef WITH_KNC
	micAvailable = _Offload_number_of_devices();
	printf("number of available MIC devices: %d\n", micAvailable);
#endif
#ifdef WITH_CUDA
	cudaGetDeviceCount(&cudaAvailable);	
	cudaProps = new cudaProp[cudaAvailable];
	for(int dev = 0; dev < cudaAvailable; ++dev){
		cudaSetDevice(dev);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		cudaProps[dev].SMXCount = deviceProp.multiProcessorCount;
		cudaProps[dev].CC = deviceProp.major * 10 + deviceProp.minor;
		printf("********CUDA device %d********\n", dev);
		printf("Device name %s\n", deviceProp.name);
		printf("Compute Capability code %d\n", cudaProps[dev].CC);
	}
	//printf("number of available CUDA devices: %d\n", cudaAvailable);
	printf("******************************\n");
#endif
	if(cudaNum == -1)
		cudaNum = cudaAvailable;	
	if(micNum == -1)
		micNum = micAvailable;
	if(cpuNum == -1)
		cpuNum = cpuAvailable;
}
