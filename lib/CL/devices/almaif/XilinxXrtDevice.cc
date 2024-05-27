/* XilinxXrtDevice.cc - Access AlmaIF device in Xilinx PCIe FPGA.

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

#include "XilinxXrtDevice.hh"

#include "AlmaifShared.hh"
#include "XilinxXrtExternalRegion.hh"
#include "XilinxXrtRegion.hh"

#include "experimental/xrt_ip.h"

#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_timing.h"
#include "xrt/xrt_bo.h"

#include <algorithm>
#include <libgen.h>

void *DeviceHandle;

XilinxXrtDevice::XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                                 unsigned j) {

  char *TmpKernelName = strdup(XrtKernelNamePrefix.c_str());
  char *KernelName = basename(TmpKernelName);

  std::string xclbin_char = XrtKernelNamePrefix + ".xclbin";

  std::string ExternalMemoryParameters =
      pocl_get_string_option("POCL_ALMAIF_EXTERNALREGION", "");

  init_xrtdevice(KernelName, xclbin_char, ExternalMemoryParameters, j);

  free(TmpKernelName);
}

XilinxXrtDevice::XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                                 const std::string &XclbinFile, unsigned j) {
  std::string ExternalMemoryParameters =
      pocl_get_string_option("POCL_ALMAIF_EXTERNALREGION", "");
  init_xrtdevice(XrtKernelNamePrefix, XclbinFile, ExternalMemoryParameters, j);
}

XilinxXrtDevice::XilinxXrtDevice(const std::string &XrtKernelNamePrefix,
                                 const std::string &XclbinFile,
                                 const std::string &ExternalMemoryParameters,
                                 unsigned j) {
  init_xrtdevice(XrtKernelNamePrefix, XclbinFile, ExternalMemoryParameters, j);
}

void XilinxXrtDevice::init_xrtdevice(
    const std::string &XrtKernelNamePrefix, const std::string &XclbinFile,
    const std::string &ExternalMemoryParameters, unsigned j) {

  auto xclEmulationMode = pocl_get_string_option("XCL_EMULATION_MODE", "");
  const char *hwEmuString = "hw_emu";
  if (!strncmp(xclEmulationMode, hwEmuString, strlen(hwEmuString))) {
    RunningOnRealFPGA_ = false;
  } else {
    RunningOnRealFPGA_ = true;
  }
  POCL_MSG_PRINT_ALMAIF("Initializing XRT device. Running on real fpga:%d\n",
                        RunningOnRealFPGA_);

  // TODO Remove magic
  size_t DeviceOffset = 0x40000000 + j * 0x10000;

  if (j == 0) {
    auto devicehandle = new xrt::device(0);
    assert(devicehandle != NULL && "devicehandle null\n");
    DeviceHandle = (void *)devicehandle;
  }

  if (pocl_exists(XclbinFile.c_str())) {
    programBitstream(XrtKernelNamePrefix, XclbinFile, j);
    ControlMemory = new XilinxXrtRegion(DeviceOffset, ALMAIF_DEFAULT_CTRL_SIZE,
                                        Kernel, DeviceOffset);

    discoverDeviceParameters();

    char TmpXclbinFile[POCL_MAX_PATHNAME_LENGTH];
    strncpy(TmpXclbinFile, XclbinFile.c_str(), POCL_MAX_PATHNAME_LENGTH);
    char *DirectoryName = dirname(TmpXclbinFile);
    std::string ImgFileName = DirectoryName;
    ImgFileName += "/" + XrtKernelNamePrefix + ".img";
    if (pocl_exists(ImgFileName.c_str())) {
      POCL_MSG_PRINT_ALMAIF(
          "Almaif: Found built-in kernel firmware. Loading it in\n");
      InstructionMemory = new XilinxXrtRegion(ImemStart, ImemSize, Kernel,
                                              ImgFileName, DeviceOffset);
    } else {
      POCL_MSG_PRINT_ALMAIF("Almaif: No default firmware found. Skipping\n");
      InstructionMemory =
          new XilinxXrtRegion(ImemStart, ImemSize, Kernel, DeviceOffset);
    }

    CQMemory = new XilinxXrtRegion(CQStart, CQSize, Kernel, DeviceOffset);
    DataMemory = new XilinxXrtRegion(DmemStart, DmemSize, Kernel, DeviceOffset);

  } else {
    // No initial xclbin. Use hardcoded layout constants derived from
    // generate_xo.tcl (local_mem_addrw_g=12, axi_addr_width_g=16,
    // axi_offset_low_g=0x40000000).
    POCL_MSG_PRINT_ALMAIF(
        "Almaif: No initial xclbin found (%s). Using hardcoded layout "
        "constants.\n",
        XclbinFile.c_str());

    // Kernel handle remains null until the first programBitstream() call.
    Kernel = nullptr;

    // Memory layout from minidebugger.vhdl with generics from generate_xo.tcl:
    //   segment_size = max(CTRL=1024, IMEM=16384, CQ=256, DMEM=16384) = 0x4000
    //   imem_offset  = DeviceOffset + 0x4000
    //   cq_offset    = DeviceOffset + 0x8000
    //   dmem_offset  = DeviceOffset + 0xC000
    PointerSize = 4;
    RelativeAddressing = false;
    ImemSize = 0x4000;
    CQSize = 0x100;
    DmemSize = 0x4000;
    ImemStart = DeviceOffset + 0x4000;
    CQStart = DeviceOffset + 0x8000;
    DmemStart = DeviceOffset + 0xC000;

    AllocRegions = (memory_region_t *)calloc(1, sizeof(memory_region_t));
    pocl_init_mem_region(AllocRegions,
                         DmemStart + ALMAIF_DEFAULT_CONSTANT_MEM_SIZE,
                         DmemSize - ALMAIF_DEFAULT_CONSTANT_MEM_SIZE);

    // Create regions with null kernel handle; handles are set via
    // setKernelPtr() from programBitstream() after the first xclbin loads.
    ControlMemory = new XilinxXrtRegion(DeviceOffset, ALMAIF_DEFAULT_CTRL_SIZE,
                                        nullptr, DeviceOffset);
    InstructionMemory =
        new XilinxXrtRegion(ImemStart, ImemSize, nullptr, DeviceOffset);
    CQMemory = new XilinxXrtRegion(CQStart, CQSize, nullptr, DeviceOffset);
    DataMemory =
        new XilinxXrtRegion(DmemStart, DmemSize, nullptr, DeviceOffset);
  }

  if (ExternalMemoryParameters != "") {
    char *tmp_params = strdup(ExternalMemoryParameters.c_str());
    char *save_ptr;
    char *param_token = strtok_r(tmp_params, ",", &save_ptr);
    size_t region_address = strtoul(param_token, NULL, 0);
    param_token = strtok_r(NULL, ",", &save_ptr);
    size_t region_size = strtoul(param_token, NULL, 0);
    if (region_size > 0) {
      ExternalXRTMemory = new XilinxXrtExternalRegion(
          region_address, region_size, DeviceHandle);
      POCL_MSG_PRINT_ALMAIF("Almaif: initialized external XRT alloc region at "
                            "%zx with size %zx\n",
                            region_address, region_size);
    }
    free(tmp_params);
  }

  PipeCount_ = 10;
  AllocatedPipes_ = (int *)calloc(PipeCount_, sizeof(int));

  XilinxXrtDeviceInitDone_ = 1;
}

XilinxXrtDevice::~XilinxXrtDevice() {
  delete ((xrt::ip *)Kernel);
  delete ((xrt::device *)DeviceHandle);
  /*  if (ExternalXRTMemory) {
      LL_DELETE(AllocRegions, AllocRegions->next);
    }*/
}

void XilinxXrtDevice::freeAndCopyExternalBuffersOutOfDevice() {
  for (auto &B : ExternalBuffers_) {
    if (hasActiveProgram_) {
      // Normal reprogram: kernel may have modified device memory, sync it
      // back.  Replace any pre-existing slot (should be null here) with a
      // fresh malloc filled from device.
      free(B.HostData);
      B.HostData = malloc(B.Size);
      assert(B.HostData);
      ExternalXRTMemory->CopyFromMMAP(B.HostData, B.P, B.Size, 0);
    }
    // else (!hasActiveProgram_): B.HostData was pre-allocated in
    // allocateBuffer() and kept up-to-date by writeDataToDevice(), so nothing
    // to do here.
    ExternalXRTMemory->freeBuffer(B.P);
  }
}

void XilinxXrtDevice::reallocateAndCopyBuffersBackToDevice() {
  for (auto &B : ExternalBuffers_) {
    ExternalXRTMemory->allocateBuffer(B.P, B.Size);
    ExternalXRTMemory->CopyToMMAP(B.P, B.HostData, B.Size, 0);
    free(B.HostData);
    B.HostData = nullptr; // xrt::bo owns the data now
  }
}

void XilinxXrtDevice::programBitstream(const std::string &XrtKernelNamePrefix,
                                       const std::string &XclbinFile,
                                       unsigned j) {

  // TODO: Fix the case when the kernel name contains a path
  // Needs to tokenize the last part of the path and use that
  // as the kernel name
  std::string XrtKernelName =
      XrtKernelNamePrefix + ":{" + XrtKernelNamePrefix + "_1}";

  uint64_t start_time = pocl_gettimemono_ns();
  unloadProgram();

  xrt::device *devicehandle = (xrt::device *)DeviceHandle;
  if (j == 0) {
    auto uuid = devicehandle->load_xclbin(XclbinFile);

    std::string MemInfo = devicehandle->get_info<xrt::info::device::memory>();
    POCL_MSG_PRINT_ALMAIF_MMAP("XRT device's memory info:%s\n",
                               MemInfo.c_str());
  }
  auto uuid = devicehandle->get_xclbin_uuid();

  POCL_MSG_PRINT_ALMAIF("Loading kernel handle\n");
  auto kernel = new xrt::ip(*devicehandle, uuid, XrtKernelName.c_str());

  assert(kernel != XRT_NULL_HANDLE &&
         "xrtKernelHandle NULL, is the kernel opened properly?");

  Kernel = (void *)kernel;

  POCL_MSG_PRINT_ALMAIF("TEST\n");
  if (XilinxXrtDeviceInitDone_) {
    ((XilinxXrtRegion *)ControlMemory)->setKernelPtr(Kernel);
    ((XilinxXrtRegion *)InstructionMemory)->setKernelPtr(Kernel);
    ((XilinxXrtRegion *)CQMemory)->setKernelPtr(Kernel);
    ((XilinxXrtRegion *)DataMemory)->setKernelPtr(Kernel);
    reallocateAndCopyBuffersBackToDevice();
  }
  hasActiveProgram_ = true;

  uint64_t end_time = pocl_gettimemono_ns();
  printf("Reprogramming done. Time: %" PRIu64 " ms\n",
         (end_time - start_time) / 1000000);
  POCL_MSG_PRINT_ALMAIF("BITSTREAM PROGRAMMING DONE\n");
}

void XilinxXrtDevice::freeBuffer(pocl_mem_identifier *P) {
  if (P->extra == 1) {
    // Only print address if the bo is real (operator bool checks the handle).
    // A sentinel default-constructed xrt::bo has no handle and no address.
    if (*(xrt::bo *)P->mem_ptr)
      POCL_MSG_PRINT_MEMORY("almaif: freed buffer from 0x%zx\n",
                            ExternalXRTMemory->pointerDeviceOffset(P));
    auto it = std::find_if(ExternalBuffers_.begin(), ExternalBuffers_.end(),
                           [P](const ExternalBuffer &B) { return B.P == P; });
    if (it != ExternalBuffers_.end()) {
      free(it->HostData);
      ExternalBuffers_.erase(it);
    }
    ExternalXRTMemory->freeBuffer(P);
  } else {
    chunk_info_t *chunk = (chunk_info_t *)P->mem_ptr;

    POCL_MSG_PRINT_MEMORY("almaif: freed buffer from 0x%zx\n",
                          chunk->start_address);

    assert(chunk != NULL);
    pocl_free_chunk(chunk);
  }
}

size_t XilinxXrtDevice::pointerDeviceOffset(pocl_mem_identifier *P) {
  if (P->extra == 1) {
    return ExternalXRTMemory->pointerDeviceOffset(P);
  } else {
    chunk_info_t *chunk = (chunk_info_t *)P->mem_ptr;
    assert(chunk != NULL);
    return chunk->start_address;
  }
}

cl_int XilinxXrtDevice::allocateBuffer(pocl_mem_identifier *P, size_t Size) {

  assert(P->mem_ptr == NULL);
  chunk_info_t *chunk = NULL;

  // TODO: add bufalloc-based on-chip memory allocation here. The current
  // version always allocates from external memory, since the current
  // kernels do not know how to access the on-chip memory.
  if (chunk == NULL) {
    if (ExternalXRTMemory) {
      // XilinxXrtExternalRegion has its own allocation requirements
      // (doesn't use bufalloc)
      P->version = 0;
      P->extra = 1;
      if (hasActiveProgram_) {
        // xclbin is loaded: allocate the device-side xrt::bo now.
        cl_int alloc_status = ExternalXRTMemory->allocateBuffer(P, Size);
        ExternalBuffers_.push_back({P, Size, nullptr});
        return alloc_status;
      } else {
        // No xclbin loaded yet.  On real HW the xrt::bo constructor requires a
        // live bitstream, so defer allocation to
        // reallocateAndCopyBuffersBackToDevice().  Store a sentinel
        // default-constructed xrt::bo so PoCL's mem_ptr != NULL assertion
        // passes; it carries no device memory and is safely destructable.
        // Keep a host-side shadow to hold writes until the first reprogram.
        P->mem_ptr = new xrt::bo();
        ExternalBuffers_.push_back({P, Size, calloc(1, Size)});
        return CL_SUCCESS;
      }
    } else {
      return CL_MEM_OBJECT_ALLOCATION_FAILURE;
    }
  } else {
    POCL_MSG_PRINT_MEMORY("almaif: allocated %zu bytes from 0x%zx\n", Size,
                          chunk->start_address);

    P->mem_ptr = chunk;
    P->extra = 0;
  }
  P->version = 0;
  return CL_SUCCESS;
}

cl_int XilinxXrtDevice::allocatePipe(pocl_mem_identifier *P, size_t Size) {

  assert(P->mem_ptr == NULL);

  P->version = 0;
  int *PipeID = (int *)calloc(1, sizeof(int));
  for (int i = 0; i < PipeCount_; i++) {
    if (AllocatedPipes_[i] == 0) {
      *PipeID = i;
      AllocatedPipes_[i] = 1;
      P->mem_ptr = PipeID;
      POCL_MSG_PRINT_MEMORY("almaif: allocated pipe %i\n", i);
      return CL_SUCCESS;
    }
  }
  return CL_MEM_OBJECT_ALLOCATION_FAILURE;
}

void XilinxXrtDevice::freePipe(pocl_mem_identifier *P) {
  int PipeID = *((int *)P->mem_ptr);
  POCL_MSG_PRINT_MEMORY("almaif: freed pipe %i\n", PipeID);
  AllocatedPipes_[PipeID] = 0;
}

int XilinxXrtDevice::pipeCount() { return PipeCount_; }

void XilinxXrtDevice::writeDataToDevice(pocl_mem_identifier *DstMemId,
                                        const char *__restrict__ const Src,
                                        size_t Size, size_t Offset) {

  if (DstMemId->extra == 0) {
    chunk_info_t *chunk = (chunk_info_t *)DstMemId->mem_ptr;
    size_t Dst = chunk->start_address + Offset;
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes to 0x%zx\n", Size, Dst);
    DataMemory->CopyToMMAP(Dst, Src, Size);
  } else if (DstMemId->extra == 1) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes to external Xrt buffer\n",
                          Size);
    if (!hasActiveProgram_) {
      // No xclbin loaded yet; no device-side xrt::bo exists.
      // Mirror the write into the pre-allocated HostData slot so
      // reallocateAndCopyBuffersBackToDevice() can push it to the device
      // once a bitstream is loaded.
      auto it = std::find_if(
          ExternalBuffers_.begin(), ExternalBuffers_.end(),
          [DstMemId](const ExternalBuffer &B) { return B.P == DstMemId; });
      if (it != ExternalBuffers_.end())
        memcpy(static_cast<char *>(it->HostData) + Offset, Src, Size);
      return; // No xrt::bo to sync to yet.
    }
    ExternalXRTMemory->CopyToMMAP(DstMemId, Src, Size, Offset);
  } else {
    POCL_ABORT("Attempt to write data to outside the device memories.\n");
  }
}

void XilinxXrtDevice::readDataFromDevice(char *__restrict__ const Dst,
                                         pocl_mem_identifier *SrcMemId,
                                         size_t Size, size_t Offset) {

  chunk_info_t *chunk = (chunk_info_t *)SrcMemId->mem_ptr;
  POCL_MSG_PRINT_ALMAIF("Reading data with chunk start %zu, and offset %zu\n",
                        chunk->start_address, Offset);
  size_t Src = chunk->start_address + Offset;
  if (SrcMemId->extra == 0) {
    POCL_MSG_PRINT_ALMAIF("almaif: Copying %zu bytes from 0x%zx\n", Size, Src);
    DataMemory->CopyFromMMAP(Dst, Src, Size);
  } else if (SrcMemId->extra == 1) {
    POCL_MSG_PRINT_ALMAIF(
        "almaif: Copying %zu bytes from external XRT buffer\n", Size);
    ExternalXRTMemory->CopyFromMMAP(Dst, SrcMemId, Size, Offset);
  } else {
    POCL_ABORT("Attempt to read data from outside the device memories.\n");
  }
}

void XilinxXrtDevice::unloadProgram() {
  if (XilinxXrtDeviceInitDone_) {
    freeAndCopyExternalBuffersOutOfDevice();
    if (hasActiveProgram_) {
      POCL_MSG_PRINT_ALMAIF("Unloading kernel handle\n");
      delete (xrt::ip *)Kernel;
      if (RunningOnRealFPGA_) {
        POCL_MSG_PRINT_ALMAIF("Unloading device handle\n");
        // Unfortunately the XRT api does not work exactly equivalently in
        // XCL_EMULATION_MODE=hw_emu and real hardware execution.
        // Recreating the devicehandle causes a crash in hw_emu,
        // but is required in the real execution.
        delete (xrt::device *)DeviceHandle;
        auto devicehandle = new xrt::device(0);
        DeviceHandle = (void *)devicehandle;
        if (ExternalXRTMemory)
          ExternalXRTMemory->setDeviceHandle(devicehandle);
      }
      hasActiveProgram_ = false;
    }
  }
}

void XilinxXrtDevice::loadProgramToDevice(almaif_kernel_data_s *KernelData,
                                          cl_kernel Kernel,
                                          _cl_command_node *Command) {
  assert(KernelData);

  char xclbin_file[POCL_MAX_PATHNAME_LENGTH];
  char img_file[POCL_MAX_PATHNAME_LENGTH];
  if (pocl_get_bool_option("POCL_MLIR_FORCE_PROGRAM_XCLBIN", false)) {
    pocl_cache_program_xclbin_path(xclbin_file, Kernel->program,
                                   Command->program_device_i);
    if (!pocl_exists(xclbin_file)) {
      POCL_ABORT("%s should have been there\n", xclbin_file);
    }
    pocl_cache_program_img_path(img_file, Kernel->program,
                                Command->program_device_i);
    if (!pocl_exists(img_file)) {
      POCL_ABORT("%s should have been there\n", img_file);
    }
  } else if (pocl_get_bool_option("POCL_MLIR_DISABLE_CMD_BUFFER_FUSION",
                                  false)) {
    pocl_cache_program_xclbin_path(
        xclbin_file,
        Kernel->higher_level_cmd_buf->fake_single_kernel_cmd_buffers[0]
            ->megaProgram,
        Command->program_device_i);
    if (!pocl_exists(xclbin_file)) {
      POCL_ABORT("%s should have been there\n", xclbin_file);
    }
    pocl_cache_program_img_path(
        img_file,
        Kernel->higher_level_cmd_buf->fake_single_kernel_cmd_buffers[0]
            ->megaProgram,
        Command->program_device_i);
    if (!pocl_exists(img_file)) {
      POCL_ABORT("%s should have been there\n", img_file);
    }
  } else {
    // first try specialized
    pocl_cache_kernel_cachedir_path(xclbin_file, Kernel->program,
                                    Command->program_device_i, Kernel,
                                    "/parallel.xclbin", Command, 1);
    if (pocl_exists(xclbin_file)) {
      pocl_cache_kernel_cachedir_path(img_file, Kernel->program,
                                      Command->program_device_i, Kernel,
                                      "/firmware.img", Command, 1);
    } else {
      // if it doesn't exist, try specialized with local sizes 0-0-0
      // should pick either 0-0-0 or 0-0-0-goffs0
      _cl_command_node cmd_copy;
      memcpy(&cmd_copy, Command, sizeof(_cl_command_node));
      cmd_copy.command.run.pc.local_size[0] = 0;
      cmd_copy.command.run.pc.local_size[1] = 0;
      cmd_copy.command.run.pc.local_size[2] = 0;

      pocl_cache_kernel_cachedir_path(xclbin_file, Kernel->program,
                                      Command->program_device_i, Kernel,
                                      "/parallel.xclbin", &cmd_copy, 1);
      if (pocl_exists(xclbin_file)) {
        pocl_cache_kernel_cachedir_path(img_file, Kernel->program,
                                        Command->program_device_i, Kernel,
                                        "/firmware.img", &cmd_copy, 1);
      } else {
        pocl_cache_kernel_cachedir_path(xclbin_file, Kernel->program,
                                        Command->program_device_i, Kernel,
                                        "/parallel.xclbin", &cmd_copy, 0);
        pocl_cache_kernel_cachedir_path(img_file, Kernel->program,
                                        Command->program_device_i, Kernel,
                                        "/firmware.img", &cmd_copy, 0);
      }
    }
  }

  assert(pocl_exists(xclbin_file));
  assert(pocl_exists(img_file));

  programBitstream("pocl_kernel", xclbin_file, 0);

  ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_RESET_CMD);

  ((XilinxXrtRegion *)InstructionMemory)->initRegion(img_file);

  for (uint32_t i = AQL_PACKET_LENGTH; i < CQMemory->Size();
       i += AQL_PACKET_LENGTH) {
    CQMemory->Write16(i, AQL_PACKET_INVALID);
  }
  CQMemory->Write32(ALMAIF_CQ_WRITE, 0);
  CQMemory->Write32(ALMAIF_CQ_READ, 0);

  ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_CONTINUE_CMD);
  HwClockStart = pocl_gettimemono_ns();
}
