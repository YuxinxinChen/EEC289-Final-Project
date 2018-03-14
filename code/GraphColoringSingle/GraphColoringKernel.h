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


/* GraphColoring,let each thread compare one host vertex's value with one of its neibor vertex */
/* if host vertex's value is smaller than its neibor, do not assign it a color */
__global__ 
void GraphColoringKernel(int numColor, uint32_t NumRow, uint32_t *col_id, uint32_t *offset, int *lbs, int *wir, int *randoms, int *colors, bool* set)
{   
for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < NumRow; i = i + blockDim.x * gridDim.x) 
 {  
    int host_id = lbs[i];
    int neibor_id = col_id[offset[host_id] + wir[i]];

    if ((colors[host_id] != 0)) return;

    int host_value = randoms[host_id];
    int neibor_value = randoms[neibor_id];
    int neibor_color = colors[neibor_id];

      if (host_id == neibor_id) return;
       if ((neibor_color != 0) && (neibor_color != numColor)) return;
        if (host_value < neibor_value)
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
  }
}
