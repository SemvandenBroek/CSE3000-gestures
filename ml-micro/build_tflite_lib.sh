#!/bin/bash
cd lib/tflite-micro/
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=cortex_m_generic TARGET_ARCH=cortex-m4 OPTIMIZED_KERNEL_DIR=cmsis_nn
cd ../../

# Patch flatbuffers utility.h to utility in file base.h


# Copy over all the downloaded libraries to the correct folder
# (These libraries are downloaded automatically by the make command on line 3)
mkdir -p lib/cortex_m_generic_cortex-m4_default/
cp lib/tflite-micro/gen/cortex_m_generic_cortex-m4_default/lib/libtensorflow-microlite.a lib/cortex_m_generic_cortex-m4_default/


