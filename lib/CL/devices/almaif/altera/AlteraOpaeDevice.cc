/* AlteraOpaeDevice.cc - Access AlmaIF device in Altera OPAE FPGA.

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

#include "AlteraOpaeDevice.hh"

#include "../AlmaifShared.hh"
#include "AlteraOpaeExternalRegion.hh"
#include "AlteraOpaeRegion.hh"

#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_timing.h"
#include <libgen.h>

#include <assert.h>
#include <opae/fpga.h>
#include <opae/mpf/mpf.h>
#include <uuid/uuid.h>

#include "aocl_mmd.h"
#include "ccip_mmd.h"
#include "ccip_mmd_device.h"
#include "fpgaconf.h"

#define N6000_PCI_OCL_BSP_AFU_ID "51ED2F4A-FEA2-4261-A595-918500575509"
const char *BITSTREAM_SUFFIX = "";

DeviceMapManager &device_manager = DeviceMapManager::get_instance();

char str_temp[1024];

fpga_handle get_mmio_handle() {
  fpga_properties filter;
  uint32_t num_matches;
  fpga_result res;
  fpga_token *tokens = nullptr;
  fpga_properties props;
  fpga_token port_token;
  fpga_handle port_handle;
  fpga_token fme_token;

  uint8_t bus;
  uint8_t device;
  uint8_t function;
  fpga_token mmio_token;
  fpga_handle mmio_handle;
  fpga_guid guid;

  fpgaGetProperties(NULL, &filter);
  fpgaPropertiesSetInterface(filter, FPGA_IFC_DFL);

  num_matches = 0;
  res = fpgaEnumerate(&filter, 1, NULL, 0, &num_matches);

  if (num_matches < 1) {
    fpgaDestroyProperties(&filter);
    printf("DFL device not found\n");
  }

  tokens = (fpga_token *)calloc(num_matches, sizeof(fpga_token));
  if (tokens == NULL) {
    printf("Error! memory not allocated.\n");
  }

  res = fpgaEnumerate(&filter, 1, tokens, num_matches, &num_matches);
  if (res != FPGA_OK) {
    fpgaDestroyProperties(&filter);
    free(tokens);
    printf("Error! enumerating.\n");
  }

  if (num_matches < 1) {
    fpgaDestroyProperties(&filter);
    free(tokens);
    printf("AFC not found\n");
  }

  for (int i = 0; i < num_matches; ++i) {
    fpgaGetProperties(tokens[i], &props);

    uint64_t oid = 0;
    fpgaPropertiesGetObjectID(props, &oid);

    uint64_t obj_id = device_manager.id_from_name(str_temp);
    if (oid == obj_id) {
      // We've found our Port..
      port_token = tokens[i];

      res = fpgaOpen(port_token, &port_handle, 0);
      if (res != FPGA_OK) {
        printf("Error opening Port: \n");
      }

      fpgaPropertiesGetBus(props, &bus);
      fpgaPropertiesGetDevice(props, &device);
      fpgaPropertiesGetFunction(props, &function);

      fpgaPropertiesGetParent(props, &fme_token);

      fpgaDestroyProperties(&props);
      break;
    }

    fpgaDestroyProperties(&props);
  }

  if (!port_token || !fme_token) {
    printf("Couldn't find tokens: \n");
  }

  fpgaGetProperties(NULL, &props);

  fpgaPropertiesSetBus(props, bus);
  fpgaPropertiesSetDevice(props, device);
  fpgaPropertiesSetInterface(props, FPGA_IFC_VFIO);

  num_matches = 0;
  res = fpgaEnumerate(&props, 1, &mmio_token, 1, &num_matches);
  if (res != FPGA_OK) {
    printf("fpgaEnumerate failed: \n");
  }

  res = fpgaPropertiesGetGUID(props, &guid);
  fpgaDestroyProperties(&props);

  if (mmio_token) {
    res = fpgaOpen(mmio_token, &mmio_handle, 0);
    if (res != FPGA_OK) {
      printf("Couldn't open mmio_token: \n");
    }
  } else
    printf("No MMIO token\n");

  free(tokens);
  return mmio_handle;
}

// Obtains accelerator handle
static fpga_handle sim_connect_to_accel(const char *accel_uuid) {
  fpga_properties filter = NULL;
  fpga_guid guid;
  fpga_token accel_token;
  uint32_t num_matches;
  fpga_handle accel_handle;
  fpga_result r;

  // Set up a filter that will search for an accelerator
  fpgaGetProperties(NULL, &filter);
  fpgaPropertiesSetObjectType(filter, FPGA_ACCELERATOR);

  // Add the desired UUID to the filter
  uuid_parse(accel_uuid, guid);
  fpgaPropertiesSetGUID(filter, guid);

  // Do the search across the available FPGA contexts
  num_matches = 1;
  fpgaEnumerate(&filter, 1, &accel_token, 1, &num_matches);

  // Not needed anymore
  fpgaDestroyProperties(&filter);

  if (num_matches < 1) {
    fprintf(stderr, "Accelerator %s not found!\n", accel_uuid);
    return 0;
  }

  // Open accelerator
  r = fpgaOpen(accel_token, &accel_handle, 0);
  assert(FPGA_OK == r);

  // Done with token
  fpgaDestroyToken(&accel_token);

  return accel_handle;
}

unsigned char *acl_loadFileIntoMemory(const char *in_file,
                                      size_t *file_size_out) {

  FILE *f = NULL;
  unsigned char *buf;
  size_t file_size;

  // When reading as binary file, no new-line translation is done.
  f = fopen(in_file, "rb");
  if (f == NULL) {
    fprintf(stderr, "Couldn't open file %s for reading\n", in_file);
    return NULL;
  }

  // get file size
  fseek(f, 0, SEEK_END);
  file_size = (size_t)ftell(f);
  rewind(f);

  // slurp the whole file into allocated buf
  buf = (unsigned char *)malloc(sizeof(char) * file_size);
  if (!buf) {
    fprintf(stderr, "Error cannot allocate memory\n");
    exit(-1);
  }
  *file_size_out = fread(buf, sizeof(char), file_size, f);
  fclose(f);

  if (*file_size_out != file_size) {
    fprintf(stderr, "Error reading %s. Read only %lu out of %lu bytes\n",
            in_file, *file_size_out, file_size);
    free(buf);
    return NULL;
  }
  return buf;
}

fpga_handle MMIOHandle;

AlteraOpaeDevice::AlteraOpaeDevice(const std::string &XrtKernelNamePrefix,
                                   unsigned j) {

  char *TmpKernelName = strdup(XrtKernelNamePrefix.c_str());
  char *KernelName = basename(TmpKernelName);

  std::string xclbin_char = XrtKernelNamePrefix + BITSTREAM_SUFFIX;

  std::string ExternalMemoryParameters =
      pocl_get_string_option("POCL_ALMAIF_EXTERNALREGION", "");

  init_opaedevice(KernelName, xclbin_char, ExternalMemoryParameters, j);

  free(TmpKernelName);
}

AlteraOpaeDevice::AlteraOpaeDevice(const std::string &XrtKernelNamePrefix,
                                   const std::string &OpaeBinFile, unsigned j) {
  std::string ExternalMemoryParameters =
      pocl_get_string_option("POCL_ALMAIF_EXTERNALREGION", "");
  init_opaedevice(XrtKernelNamePrefix, OpaeBinFile, ExternalMemoryParameters,
                  j);
}

AlteraOpaeDevice::AlteraOpaeDevice(const std::string &XrtKernelNamePrefix,
                                   const std::string &OpaeBinFile,
                                   const std::string &ExternalMemoryParameters,
                                   unsigned j) {
  init_opaedevice(XrtKernelNamePrefix, OpaeBinFile, ExternalMemoryParameters,
                  j);
}

void AlteraOpaeDevice::init_opaedevice(
    const std::string &XrtKernelNamePrefix, const std::string &OpaeBinFile,
    const std::string &ExternalMemoryParameters, unsigned j) {

  POCL_MSG_PRINT_ALMAIF("Initializing Opae device\n");

  // auto devicehandle = new xrt::device(0);
  // assert(devicehandle != NULL && "devicehandle null\n");
  // DeviceHandle = (void *)devicehandle;
  programBitstream(XrtKernelNamePrefix, OpaeBinFile, j);
  // TODO Remove magic
  size_t DeviceOffset = 0x40000000 + j * 0x10000;
  // size_t DeviceOffset = 0x00000000;
  ControlMemory = new AlteraOpaeRegion(DeviceOffset, ALMAIF_DEFAULT_CTRL_SIZE,
                                       MMIOHandle, DeviceOffset);

  discoverDeviceParameters();

  char TmpOpaeBinFile[POCL_MAX_PATHNAME_LENGTH];
  strncpy(TmpOpaeBinFile, OpaeBinFile.c_str(), POCL_MAX_PATHNAME_LENGTH);
  char *DirectoryName = dirname(TmpOpaeBinFile);
  std::string ImgFileName = DirectoryName;
  ImgFileName += "/" + XrtKernelNamePrefix + ".img";
  if (pocl_exists(ImgFileName.c_str())) {
    POCL_MSG_PRINT_ALMAIF(
        "Almaif: Found built-in kernel firmware. Loading it in\n");
    InstructionMemory = new AlteraOpaeRegion(ImemStart, ImemSize, MMIOHandle,
                                             ImgFileName, DeviceOffset);
  } else {
    POCL_MSG_PRINT_ALMAIF("Almaif: No default firmware found. Skipping\n");
    InstructionMemory =
        new AlteraOpaeRegion(ImemStart, ImemSize, MMIOHandle, DeviceOffset);
  }

  CQMemory = new AlteraOpaeRegion(CQStart, CQSize, MMIOHandle, DeviceOffset);
  DataMemory =
      new AlteraOpaeRegion(DmemStart, DmemSize, MMIOHandle, DeviceOffset);

  if (ExternalMemoryParameters != "") {
    char *tmp_params = strdup(ExternalMemoryParameters.c_str());
    char *save_ptr;
    char *param_token = strtok_r(tmp_params, ",", &save_ptr);
    size_t region_address = strtoul(param_token, NULL, 0);
    param_token = strtok_r(NULL, ",", &save_ptr);
    size_t region_size = strtoul(param_token, NULL, 0);
    if (region_size > 0) {
      ExternalOpaeMemory =
          new AlteraOpaeExternalRegion(region_address, region_size, MMIOHandle);
      POCL_MSG_PRINT_ALMAIF("Almaif: initialized external XRT alloc region at "
                            "%zx with size %zx\n",
                            region_address, region_size);
    }
    free(tmp_params);
  }

  AlteraOpaeDeviceInitDone_ = 1;
}

AlteraOpaeDevice::~AlteraOpaeDevice() {
  unloadProgram();
  /*  if (ExternalOpaeMemory) {
      LL_DELETE(AllocRegions, AllocRegions->next);
    }*/
}

/*
void AlteraOpaeDevice::freeAndCopyExternalBuffersOutOfDevice() {
  for (int i = 0; i < AllocatedExternalBuffers_.size(); i++) {
    pocl_mem_identifier *P = AllocatedExternalBuffers_[i].first;
    size_t Size = AllocatedExternalBuffers_[i].second;

    void *temporary_buffer = malloc(Size);
    assert(temporary_buffer);
    ExternalOpaeMemory->CopyFromMMAP(temporary_buffer, P, Size, 0);
    ExternalOpaeMemory->freeBuffer(P);

    ExternalBufferData_.push_back(temporary_buffer);
  }
}

void AlteraOpaeDevice::reallocateAndCopyBuffersBackToDevice() {
  for (int i = 0; i < AllocatedExternalBuffers_.size(); i++) {
    pocl_mem_identifier *P = AllocatedExternalBuffers_[i].first;
    size_t Size = AllocatedExternalBuffers_[i].second;

    ExternalOpaeMemory->allocateBuffer(P, Size);
    ExternalOpaeMemory->CopyToMMAP(P, ExternalBufferData_[i], Size, 0);
    free(ExternalBufferData_[i]);
  }
  ExternalBufferData_.clear();
}
*/

void AlteraOpaeDevice::programBitstream(const std::string &XrtKernelNamePrefix,
                                        const std::string &OpaeBinFile,
                                        unsigned j) {
  uint64_t start_time = pocl_gettimemono_ns();

  if (!pocl_get_bool_option("POCL_OPAE_SIM", false)) {

    unloadProgram();

    int aocl_ret;
    std::string fpga_conf = "fpgaconf -V -B 0x86 " + OpaeBinFile + ".gbs";
    system(fpga_conf.c_str());

    aocl_ret = aocl_mmd_get_offline_info(AOCL_MMD_BOARD_NAMES, sizeof(str_temp),
                                         str_temp, NULL);
    if (aocl_ret < 0)
      printf("Error getting offline board name\n");
    // else printf("Board name is %s\n",str_temp);

    size_t file_size;
    unsigned char *bin_file = NULL;
    bin_file =
        acl_loadFileIntoMemory((OpaeBinFile + ".bin").c_str(), &file_size);

    int DeviceHandle = aocl_mmd_open(str_temp);
    if (DeviceHandle < 0)
      printf("Error getting handle\n");
    // else printf("Board handle obtained\n");

    aocl_ret = aocl_mmd_program(DeviceHandle, bin_file, file_size,
                                AOCL_MMD_PROGRAM_PRESERVE_GLOBAL_MEM);
    if (aocl_ret < 0)
      printf("Failed in programming device\n");
    else
      printf("Device pogrammed.\n");

    aocl_ret = aocl_mmd_close(DeviceHandle);
    if (aocl_ret < 0)
      printf("Device not closed\n");
    // else printf("Device closed\n");

    MMIOHandle = get_mmio_handle();

    POCL_MSG_PRINT_ALMAIF("TEST\n");
    if (AlteraOpaeDeviceInitDone_) {
      ((AlteraOpaeRegion *)ControlMemory)->setKernelPtr(MMIOHandle);
      ((AlteraOpaeRegion *)InstructionMemory)->setKernelPtr(MMIOHandle);
      ((AlteraOpaeRegion *)CQMemory)->setKernelPtr(MMIOHandle);
      ((AlteraOpaeRegion *)DataMemory)->setKernelPtr(MMIOHandle);
      // reallocateAndCopyBuffersBackToDevice();
    }

    uint64_t end_time = pocl_gettimemono_ns();
    printf("Reprogramming done. Time: %" PRIu64 " ms\n",
           (end_time - start_time) / 1000000);
    POCL_MSG_PRINT_ALMAIF("BITSTREAM PROGRAMMING DONE\n");
  } else {

    MMIOHandle = sim_connect_to_accel(N6000_PCI_OCL_BSP_AFU_ID);
  }
  hasActiveProgram_ = true;
}

void AlteraOpaeDevice::freeBuffer(pocl_mem_identifier *P) {
  if (P->extra == 1) {
    POCL_MSG_PRINT_MEMORY("almaif: freed buffer from 0x%zx\n",
                          ExternalOpaeMemory->pointerDeviceOffset(P));
    /*    auto newEnd = std::remove_if(
            AllocatedExternalBuffers_.begin(), AllocatedExternalBuffers_.end(),
            [&P](const std::pair<pocl_mem_identifier *, size_t> &elem) {
              return elem.first == P;
            });
        AllocatedExternalBuffers_.erase(newEnd,
       AllocatedExternalBuffers_.end());
        */
    ExternalOpaeMemory->freeBuffer(P);
  } else {
    chunk_info_t *chunk = (chunk_info_t *)P->mem_ptr;

    POCL_MSG_PRINT_MEMORY("almaif: freed buffer from 0x%zx\n",
                          chunk->start_address);

    assert(chunk != NULL);
    pocl_free_chunk(chunk);
  }
}

size_t AlteraOpaeDevice::pointerDeviceOffset(pocl_mem_identifier *P) {
  if (P->extra == 1) {
    return ExternalOpaeMemory->pointerDeviceOffset(P);
  } else {
    chunk_info_t *chunk = (chunk_info_t *)P->mem_ptr;
    assert(chunk != NULL);
    return chunk->start_address;
  }
}

cl_int AlteraOpaeDevice::allocateBuffer(pocl_mem_identifier *P, size_t Size) {

  assert(P->mem_ptr == NULL);
  chunk_info_t *chunk = NULL;

  // TODO: add bufalloc-based on-chip memory allocation here. The current
  // version always allocates from external memory, since the current
  // kernels do not know how to access the on-chip memory.
  if (chunk == NULL) {
    if (ExternalOpaeMemory) {
      // AlteraOpaeExternalRegion has its own allocation requirements
      // (doesn't use bufalloc)
      cl_int alloc_status = ExternalOpaeMemory->allocateBuffer(P, Size);
      P->version = 0;
      P->extra = 1;
      // AllocatedExternalBuffers_.push_back({P, Size});
      return alloc_status;
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

cl_int AlteraOpaeDevice::allocatePipe(pocl_mem_identifier *P, size_t Size) {

  return CL_MEM_OBJECT_ALLOCATION_FAILURE;
}

void AlteraOpaeDevice::freePipe(pocl_mem_identifier *P) {}

int AlteraOpaeDevice::pipeCount() { return 0; }

void AlteraOpaeDevice::writeDataToDevice(pocl_mem_identifier *DstMemId,
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
    ExternalOpaeMemory->CopyToMMAP(DstMemId, (char *__restrict__)Src, Size,
                                   Offset);
  } else {
    POCL_ABORT("Attempt to write data to outside the device memories.\n");
  }
}

void AlteraOpaeDevice::readDataFromDevice(char *__restrict__ const Dst,
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
    ExternalOpaeMemory->CopyFromMMAP(Dst, SrcMemId, Size, Offset);
  } else {
    POCL_ABORT("Attempt to read data from outside the device memories.\n");
  }
}

void AlteraOpaeDevice::unloadProgram() {
  if (hasActiveProgram_) {
    //    ExternalBufferData_.clear();
    if (AlteraOpaeDeviceInitDone_) {
      // freeAndCopyExternalBuffersOutOfDevice();
      POCL_MSG_PRINT_ALMAIF("Unloading kernel handle\n");
      //     delete (xrt::ip *)Kernel;
      if (!pocl_get_bool_option("POCL_OPAE_SIM", false)) {
        fpga_result res = fpgaClose(MMIOHandle);
        if (res != FPGA_OK)
          printf("Device not closed\n");

        POCL_MSG_PRINT_ALMAIF("Unloading device handle\n");
        // Unfortunately the XRT api does not work exactly equivalently in
        // XCL_EMULATION_MODE=hw_emu and real hardware execution.
        // Recreating the devicehandle causes a crash in hw_emu,
        // but is required in the real execution.
        // delete (xrt::device *)DeviceHandle;
        // auto devicehandle = new xrt::device(0);
        // DeviceHandle = (void *)devicehandle;
        // ExternalOpaeMemory->setDeviceHandle(devicehandle);
      }
    }
    hasActiveProgram_ = false;
  }
}

void AlteraOpaeDevice::loadProgramToDevice(almaif_kernel_data_s *KernelData,
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
    pocl_cache_kernel_cachedir_path(
        xclbin_file, Kernel->program, Command->program_device_i, Kernel,
        std::string("/parallel" + std::string(BITSTREAM_SUFFIX)).c_str(),
        Command, 1);
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

      pocl_cache_kernel_cachedir_path(
          xclbin_file, Kernel->program, Command->program_device_i, Kernel,
          std::string("/parallel" + std::string(BITSTREAM_SUFFIX)).c_str(),
          &cmd_copy, 1);
      if (pocl_exists(xclbin_file)) {
        pocl_cache_kernel_cachedir_path(img_file, Kernel->program,
                                        Command->program_device_i, Kernel,
                                        "/firmware.img", &cmd_copy, 1);
      } else {
        pocl_cache_kernel_cachedir_path(
            xclbin_file, Kernel->program, Command->program_device_i, Kernel,
            std::string("/parallel" + std::string(BITSTREAM_SUFFIX)).c_str(),
            &cmd_copy, 0);
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

  InstructionMemory->initRegion(img_file);

  for (uint32_t i = AQL_PACKET_LENGTH; i < CQMemory->Size();
       i += AQL_PACKET_LENGTH) {
    CQMemory->Write16(i, AQL_PACKET_INVALID);
  }
  CQMemory->Write32(ALMAIF_CQ_WRITE, 0);
  CQMemory->Write32(ALMAIF_CQ_READ, 0);

  ControlMemory->Write32(ALMAIF_CONTROL_REG_COMMAND, ALMAIF_CONTINUE_CMD);
  HwClockStart = pocl_gettimemono_ns();
}
