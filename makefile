# Name of binary file
BIN=histogram_equalizer
# Flags
FLAGS=-Wall -O3
FLAGS_NVCC=-gencode=arch=compute_60,code=sm_60 
# Includes
INC=-I./include -I/usr/local/cuda/include
# Sources directory
SRC=src
# Librairies directory
LIBDIR=
# Librairies
LIBS=-lcuda -lcudart
# Object Directory
OBJDIR=obj
# Compilers
NVCC=nvcc
GPP=g++
# Object files to genereate from C++
OBJECTS= $(OBJDIR)/lodepng.o $(OBJDIR)/common.o $(OBJDIR)/main.o
# Object files to generate from CUDA (.cu)
OBJECTS_CUDA=$(OBJDIR)/HistogramEqualizer.cu.o

all: $(OBJDIR) $(OBJECTS) $(OBJECTS_CUDA)
	@echo "**** LINKING ****"
	$(NVCC) $(LIBDIR) $(LIBS) $(FLAGS_NVCC) $(OBJECTS) $(OBJECTS_CUDA) -o $(BIN)

$(OBJDIR):
	mkdir -p $@

$(OBJDIR)/%.o: $(SRC)/%.cpp
	@echo "**** $@ ****"
	$(GPP) $(INC) $(FLAGS) -c $< -o $@

$(OBJDIR)/%.cu.o: $(SRC)/%.cu
	@echo "**** $@ ****"
	$(NVCC) $(FLAGS_NVCC) $(INC) -c $< -o $@

clean:
	rm -r $(OBJDIR)
	rm $(BIN)

