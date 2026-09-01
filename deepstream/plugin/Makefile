CUDA_VER ?= 12.6
CXX := g++
CXXFLAGS := -Wall -Wextra -std=c++17 -fPIC \
  -I/opt/nvidia/deepstream/deepstream/sources/includes \
  -I/usr/local/cuda-$(CUDA_VER)/include
LDFLAGS := -shared -L/usr/local/cuda-$(CUDA_VER)/lib64 -lnvinfer -lcudart
TARGET := libnvdsinfer_custom_impl_Rune_pose.so

all: $(TARGET)

$(TARGET): nvdsparsepose_Rune.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
