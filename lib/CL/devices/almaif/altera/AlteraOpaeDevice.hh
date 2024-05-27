/* AlteraOpaeDevice.hh - Access AlmaIF device in Altera OPAE FPGA.

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

#ifndef ALTERAOPAEDEVICE_H
#define ALTERAOPAEDEVICE_H

#include "../AlmaIFDevice.hh"

class AlteraOpaeExternalRegion;

// This class abstracts the Almaif device instantiated on a Xilinx (PCIe) FPGA.
// The FPGA is reconfigured and Almaif device's memory map accessed with
// the Xilinx Runtime (XRT) API.
class AlteraOpaeDevice : public AlmaIFDevice {
public:
  AlteraOpaeDevice(const std::string &XrtKernelNamePrefix, unsigned j);
  AlteraOpaeDevice(const std::string &XrtKernelNamePrefix,
                   const std::string &OpaeBinFile, unsigned j);
  AlteraOpaeDevice(const std::string &XrtKernelNamePrefix,
                   const std::string &OpaeBinFile,
                   const std::string &ExternalMemoryParameters, unsigned j);
  void init_opaedevice(const std::string &XrtKernelNamePrefix,
                       const std::string &OpaeBinFile,
                       const std::string &ExternalMemoryParameters, unsigned j);
  ~AlteraOpaeDevice() override;
  void loadProgramToDevice(almaif_kernel_data_s *KernelData, cl_kernel Kernel,
                           _cl_command_node *Command) override;
  void unloadProgram() override;
  // Reconfigures the FPGA
  void programBitstream(const std::string &XrtKernelNamePrefix,
                        const std::string &OpaeBinFile, unsigned j);

  // Allocate buffers from either on-chip or external memory regions
  // (Directs to either AlteraOpaeRegion or AlteraOpaeExternalRegion)
  cl_int allocateBuffer(pocl_mem_identifier *P, size_t Size) override;
  void freeBuffer(pocl_mem_identifier *P) override;
  // Retuns the offset of the allocated buffer, in order to be passed
  // as a kernel argument. This is relevant for AlteraOpaeDevice specifically,
  // since the allocations in AlteraOpaeExternalRegion are managed by XRT API.
  size_t pointerDeviceOffset(pocl_mem_identifier *P) override;
  void writeDataToDevice(pocl_mem_identifier *DstMemId,
                         const char *__restrict__ const Src, size_t Size,
                         size_t Offset) override;
  void readDataFromDevice(char *__restrict__ const Dst,
                          pocl_mem_identifier *SrcMemId, size_t Size,
                          size_t Offset) override;

  // unimplemented
  cl_int allocatePipe(pocl_mem_identifier *P, size_t Size) override;
  void freePipe(pocl_mem_identifier *P) override;
  int pipeCount() override;

private:
  AlteraOpaeExternalRegion *ExternalOpaeMemory;
  void *Kernel;
  int AlteraOpaeDeviceInitDone_ = 0;
  bool hasActiveProgram_ = false;
};

#endif
