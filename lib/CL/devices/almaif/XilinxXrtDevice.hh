/* XilinxXrtDevice.hh - Access AlmaIF device in Xilinx PCIe FPGA.

   Copyright (c) 2022 Topi Leppänen / Tampere University

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

#ifndef XILINXXRTDEVICE_H
#define XILINXXRTDEVICE_H

#include "AlmaIFDevice.hh"

#include <vector>

class XilinxXrtExternalRegion;

// This class abstracts the Almaif device instantiated on a Xilinx (PCIe) FPGA.
// The FPGA is reconfigured and Almaif device's memory map accessed with
// the Xilinx Runtime (XRT) API.
class XilinxXrtDevice : public AlmaIFDevice {
public:
  XilinxXrtDevice(const std::string &XrtKernelNamePrefix, unsigned j);
  XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                  const std::string &XclbinFile, unsigned j);
  XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                  const std::string &XclbinFile,
                  const std::string &ExternalMemoryParameters, unsigned j);
  void init_xrtdevice(const std::string &XrtKernelNamePrefix,
                      const std::string &XclbinFile,
                      const std::string &ExternalMemoryParameters, unsigned j);
  ~XilinxXrtDevice() override;
  void loadProgramToDevice(almaif_kernel_data_s *KernelData, cl_kernel Kernel,
                           _cl_command_node *Command) override;
  void unloadProgram() override;
  // Reconfigures the FPGA
  void programBitstream(const std::string &XrtKernelNamePrefix,
                        const std::string &XclbinFile, unsigned j);
  bool isHardwareReady() override { return Kernel != nullptr; }

  // Allocate buffers from either on-chip or external memory regions
  // (Directs to either XilinxXrtRegion or XilinxXrtExternalRegion)
  cl_int allocateBuffer(pocl_mem_identifier *P, size_t Size) override;
  void freeBuffer(pocl_mem_identifier *P) override;
  // Retuns the offset of the allocated buffer, in order to be passed
  // as a kernel argument. This is relevant for XilinxXrtDevice specifically,
  // since the allocations in XilinxXrtExternalRegion are managed by XRT API.
  size_t pointerDeviceOffset(pocl_mem_identifier *P) override;
  void writeDataToDevice(pocl_mem_identifier *DstMemId,
                         const char *__restrict__ const Src, size_t Size,
                         size_t Offset) override;
  void readDataFromDevice(char *__restrict__ const Dst,
                          pocl_mem_identifier *SrcMemId, size_t Size,
                          size_t Offset) override;
  cl_int allocatePipe(pocl_mem_identifier *P, size_t Size) override;
  void freePipe(pocl_mem_identifier *P) override;
  int pipeCount() override;

private:
  XilinxXrtExternalRegion *ExternalXRTMemory = nullptr;
  void *Kernel;
  int XilinxXrtDeviceInitDone_ = 0;
  int PipeCount_ = 0;
  int *AllocatedPipes_;
  bool hasActiveProgram_ = false;
  bool RunningOnRealFPGA_ = true;
  struct ExternalBuffer {
    pocl_mem_identifier *P;
    size_t Size;
    // Host-side shadow copy.  Pre-allocated (calloc) when no xclbin is loaded
    // yet so that writeDataToDevice() can mirror writes and
    // freeAndCopyExternalBuffersOutOfDevice() can recover data without syncing
    // from the device.  Filled by CopyFromMMAP on normal reprogram paths.
    // Null while the xrt::bo owns the data.
    void *HostData;
  };
  std::vector<ExternalBuffer> ExternalBuffers_;
  void freeAndCopyExternalBuffersOutOfDevice();
  void reallocateAndCopyBuffersBackToDevice();
};

#endif
