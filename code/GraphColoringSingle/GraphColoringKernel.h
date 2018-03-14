#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>

using namespace mgpu;

/* find each neibor vertex's rank in the host vertex's list */
__global__
void WorkItemRank(int *scan, int *lbs, int *wir, int sizeLbs) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeLbs; i+=blockDim.x*gridDim.x)
	{
		wir[i] = i - scan[lbs[i]];
	}
}

/* GraphColoring,let each thread compare one host vertex's value with one of its neighbor vertexes */
/* if host vertex's value is smaller than its neibor, do not assign it a color */
__global__ 
void GraphColoringKernel(int numColor,int *col_id, int *offset, int *bls, int *wir, int *randoms, int *colors)
{   
for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < V; i = i + blockDim.x * gridDim.x) 
 {  
    int host_id = lbs[i];
    int neibor_id = col_id[offset[host_id] + wir[i]];

    if ((colors[host_id] != 0)) continue;

    int host_value = randoms[host_id];
    int neibor_value = randoms[neibor_id];
    int neibor_color = colors[neibor_id];

      if (((neibor_color != 0) && (neibor_color != numColor)) || (host_id == neibor_id)) continue;
        if (host_value < neibor_value) continue;
         colors[host_id] = numColor;
 }
}
