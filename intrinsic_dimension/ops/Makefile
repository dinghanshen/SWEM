TF_INC := $(shell python2 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

all: zero_out.so fast_walsh_hadamard.so fast_walsh_hadamard.cu.o

zero_out.so: zero_out.cc
	g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I ${TF_INC} -O2 -D_GLIBCXX_USE_CXX11_ABI=0

fast_walsh_hadamard.so: fast_walsh_hadamard.cc fast_walsh_hadamard.cu.o
	g++ -std=c++11 -shared fast_walsh_hadamard.cc fast_walsh_hadamard.cu.o -o fast_walsh_hadamard.so -fPIC -I ${TF_INC} -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -D GOOGLE_CUDA=1 -lcudart

fast_walsh_hadamard.cu.o: fast_walsh_hadamard.cu.cc
	nvcc -std=c++11 -c -o fast_walsh_hadamard.cu.o fast_walsh_hadamard.cu.cc -I ${TF_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -D_MWAITXINTRIN_H_INCLUDED --gpu-architecture=compute_53 --gpu-code="sm_53,compute_53"

clean:
	rm -f zero_out.so fast_walsh_hadamard.so fast_walsh_hadamard.cu.o
