#!/bin/bash
cd tflite-micro/
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4 OPTIMIZED_KERNEL_DIR=cmsis_nn

# Patch flatbuffers utility.h to utility in file base.h
# Copy over all the libraries to the correct folder