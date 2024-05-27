/* AlmaifCompileMLIR.hh - compiler support for HLS Almaif devices

   Copyright (c) 2025 Topi Leppänen / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#ifndef POCL_ALMAIFCOMPILEMLIR_H
#define POCL_ALMAIFCOMPILEMLIR_H

#include "CL/cl.h"
#include "pocl.h"
#include "pocl_threads.h"
#include <string>

int pocl_almaif_mlir_initialize(cl_device_id device,
                                const std::string &parameters);
int pocl_almaif_mlir_cleanup(cl_device_id device);
int pocl_almaif_mlir_compile(_cl_command_node *cmd, cl_kernel kernel,
                             cl_device_id device, int specialize);
int pocl_almaif_mlir_compile_kernels(int num_kernels, _cl_command_node **cmds,
                                     cl_kernel *kernels, cl_device_id device,
                                     int specialize);

char *pocl_almaif_mlir_init_build(void *data);

int pocl_almaif_mlir_build_source(cl_program program, cl_uint device_i,
                                  cl_uint num_input_headers,
                                  const cl_program *input_headers,
                                  const char **header_include_names,
                                  int link_program);

typedef struct mlir_backend_data_s {
  pocl_lock_t mlir_compile_lock
      __attribute__((aligned(HOST_CPU_CACHELINE_SIZE)));
  std::string machine_file;
  int core_count;
} mlir_backend_data_t;

void pocl_mlir_write_kernel_descriptor(char *content, size_t content_size,
                                       _cl_command_node *command,
                                       cl_kernel kernel, cl_device_id device,
                                       int specialize);

int pocl_almaif_mlir_device_hash(const char *adf_file, const char *llvm_triplet,
                                 char *output);

int pocl_almaif_mlir_setup_metadata(cl_device_id device, cl_program program,
                                    unsigned program_device_i);

cl_int pocl_almaif_mlir_create_finalized_command_buffer(
    cl_device_id device, cl_command_buffer_khr command_buffer);
cl_int
pocl_almaif_mlir_free_command_buffer(cl_device_id device,
                                     cl_command_buffer_khr command_buffer);
cl_int pocl_almaif_mlir_run_command_buffer(cl_device_id device,
                                           cl_command_buffer_khr cmd,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event *event_wait_list,
                                           cl_event *event);

#endif
