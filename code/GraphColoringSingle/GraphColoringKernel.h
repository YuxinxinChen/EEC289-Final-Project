#include <moderngpu/transform.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>

using namespace mgpu;

/* find each neibor vertex's rank in the host vertex's list */
__global__
void WorkItemRank(int *scan, int *lbs, int sizeLbs, uint32_t *col_id, uint32_t *offset, int *neighbor) {
	for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeLbs; i+=blockDim.x*gridDim.x)
	{
		int wir = i - scan[lbs[i]];
		neighbor[i] = col_id[offset[lbs[i]] + wir];
	}
}


/* GraphColoring,let each thread compare one host vertex's value with one of its neibor vertex */
/* if host vertex's value is smaller than its neibor, do not assign it a color */
__global__ 
void GraphColoringKernel(int numColor, uint32_t numNNZ, uint32_t *col_id, uint32_t *offset, int *lbs, int *wir,  int *colors, bool* set)
{   
for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numNNZ; i = i + blockDim.x * gridDim.x) 
 {  
    int host_id = lbs[i];
    int neibor_id = col_id[offset[host_id] + wir[i]];
    
    if(set[host_id]==0) continue;
    if ((colors[host_id] != 0)) continue;
    if(colors[neibor_id] != 0) continue;


    if (host_id < neibor_id)
    {
       set[host_id] = false;
    }
 }
}

/* the setTrue is initiated to be true
if the vertex won't be colored or has already been colored, the setTrue is false
else the setTrue is true*/
__global__
void ColorChanging(int numColor, uint32_t NumRow, int* colors, bool* setTrue)
{
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < NumRow; i = i + blockDim.x * gridDim.x)
  {
    if ((colors[i] != 0)) continue;
    if (setTrue[i]) 
      {
        colors[i] = numColor;
      }
    setTrue[i]=1;
  }
}

__global__
void FindChangeColor(int *lbs, int sizeLbs, int *neighbor, int currentColor, int *color, bool *set) {
        for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<sizeLbs; i+=blockDim.x*gridDim.x)
        {
                int neighborOwner = lbs[i];
                int neigh = neighbor[i];

                if(neigh >= neighborOwner && (color[neigh]==0 || color[neigh]==currentColor))
                        set[neighborOwner]=0;

        }
}

__global__
void assignColor(bool *set, int numNodes, int *color, int currentColor)
{
        for(int i=blockIdx.x*blockDim.x+threadIdx.x; i<numNodes; i=i+gridDim.x*blockDim.x)
        {
                if(set[i]==1 && color[i]==0)
                        color[i]=currentColor;
                set[i]=1;
        }
}
