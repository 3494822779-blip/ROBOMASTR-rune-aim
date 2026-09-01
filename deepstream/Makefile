CUDA_VER ?= 12.6

CC:= gcc
CFLAGS:= -Wall -Wfatal-errors -fPIC -Wno-deprecated-declarations
APP:= bin/deepstream
BUILD_DIR:= build/app
DS_SDK_ROOT:= /opt/nvidia/deepstream/deepstream
LIB_INSTALL_DIR?= $(DS_SDK_ROOT)/lib/

TARGET_DEVICE:= $(shell gcc -dumpmachine | cut -f1 -d -)
ifeq ($(TARGET_DEVICE), aarch64)
	CFLAGS+= -DPLATFORM_TEGRA
endif

SRCS:= src/deepstream.c src/modules/interrupt.c src/modules/perf.c
OBJS:= $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SRCS))
INCS:= $(wildcard src/*.h) $(wildcard src/modules/*.h)
PKGS:= gstreamer-1.0

CFLAGS+= -Isrc -I$(DS_SDK_ROOT)/sources/apps/apps-common/includes \
	-I$(DS_SDK_ROOT)/sources/includes -I/usr/local/cuda-$(CUDA_VER)/include
CFLAGS+= `pkg-config --cflags $(PKGS)`
LIBS:= `pkg-config --libs $(PKGS)`
LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper \
	-L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lcuda \
	-Wl,-rpath,$(LIB_INSTALL_DIR)

.PHONY: all plugin clean

all: plugin $(APP)

plugin:
	$(MAKE) -C plugin CUDA_VER=$(CUDA_VER)

$(BUILD_DIR)/%.o: src/%.c $(INCS) Makefile
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	@mkdir -p $(dir $@)
	$(CC) -o $@ $(OBJS) $(LIBS)

clean:
	rm -rf $(BUILD_DIR) $(APP)
	$(MAKE) -C plugin clean CUDA_VER=$(CUDA_VER)
