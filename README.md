# CUDAProject
A CUDA implementation about image treatment.

Members :
- CAQUELARD Vincent
- GBOHO Thierry
- HEBRAS Jérôme

Compilation :
1. $ cd build
2. $ cmake ..
3. $ make
  
Execution :  
gpu_img_processor \[in_path] \[out_path] \[dimX] \[dimY] \[useSharedMemory] \[useNumberStream] \[filter pass]  
  
in_path : /path/to/image/in  
out_path : /path/to/image/out  
dimX : integer  
dimY : integer  
useSharedMemory : 1 = true, 0 = false  
useNumberStream : integer representing the number of streams to use. (0 means no stream).  
filter :  
- boxblur
- gaussianblur
- gaussianblur5
- emboss
- emboss5
- sharpen
pass : how many times the filter have to be applied.  
The parameter \[filter pass] can be repeated how many times you want to.
