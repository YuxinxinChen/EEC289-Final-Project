__global__ void stopKernel(int *color, int *shouldStop, int NodeNum)
{
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<NodeNum; i=i+gridDim.x*blockDim.x)
	{
		if(color[i] == 0)
			*shouldStop = 0;
	}
}
	
