# CUDAProject
### A CUDA implementation about image treatment.

#### Members :
- CAQUELARD Vincent
- GBOHO Thierry
- HEBRAS Jérôme

#### Compilation :
1. $ cd build
2. $ cmake ..
3. $ make
  
#### Execution :  
**gpu_img_processor** \[in_path="in.jpg"] \[out_path="out.jpg"] \[dimX=32] \[dimY=32] \[useSharedMemory=0] \[useNumberStream=0] \[filter="boxblur" pass=1]  

*\[parameter_name=default_value]*
  
**in_path** : /path/to/image/in  
**out_path** : /path/to/image/out  
**dimX** : integer  
**dimY** : integer  
**useSharedMemory** : 1 = true, 0 = false  
**useNumberStream** : integer representing the number of streams to use. (0 means no stream).  
**filter** :  
- boxblur
- gaussianblur
- gaussianblur5
- emboss
- emboss5
- sharpen

**pass** : how many times the filter will be applied.  
*Note : The parameter \[filter pass] can be repeated how many times you want to.*
