/* AlteraOpaeExternalRegion.cc - Access external memory (DDR or HBM) of an OPAE
 device
 *                        as AlmaIFRegion

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

#include <assert.h>
#include <stdlib.h>
#include <unistd.h>

#include "AlteraOpaeExternalRegion.hh"

#include <unordered_map>

#include <assert.h>
#include <opae/fpga.h>
#include <opae/mpf/mpf.h>
#include <uuid/uuid.h>

#include "aocl_mmd.h"
#include "ccip_mmd.h"
#include "ccip_mmd_device.h"
#include "fpgaconf.h"

using namespace std;
std::unordered_map<void *, uint64_t> pinned_mem;

#define SVM_CCIP_MMD_MPF 0x24000

const uint64_t KB = 2 << 9;
const uint64_t MB = 2 << 20;

const uint64_t buf_threshold = 1 * MB;
const uint64_t dma_buffer_sz = 2 * MB;

void *dma_buffer_write = nullptr;
void *dma_buffer_read = nullptr;
const int TIMEOUT = 10000;

// Pins memory if transfer is greater than 4KB and 2MB
int AlteraOpaeExternalRegion::pinning_memory(void *dma_host_addr, size_t len) {

  fpga_result res = FPGA_OK;

  uint64_t ioaddr = 0;
  uint64_t span;
  uint64_t addr = reinterpret_cast<uint64_t>(dma_host_addr);
  const uint64_t page_sz = 1 << 12;
  const uint64_t page_mask = ~((page_sz)-1);
  uint64_t start_page = addr & page_mask;
  uint64_t end_page = (addr + len) & page_mask;
  if (start_page == addr && end_page == (addr + len)) {
    span = end_page - start_page;
  } else {
    span = end_page - start_page + page_sz;
  }
  void *host_addr = reinterpret_cast<void *>(start_page);
  mpf_vtp_page_size page_size;

  // checking if addr already pinned for 4KB page size
  page_size = MPF_VTP_PAGE_4KB;
  res = mpfVtpPinAndGetIOAddress(MPFHandle_, MPF_VTP_PIN_MODE_LOOKUP_ONLY,
                                 host_addr, &ioaddr, &page_size, nullptr);
  if (res == FPGA_OK) {
    printf("Page size is 4KB\n");
    return 2;
  }

  page_size = MPF_VTP_PAGE_2MB;
  res = mpfVtpPinAndGetIOAddress(MPFHandle_, MPF_VTP_PIN_MODE_LOOKUP_ONLY,
                                 host_addr, &ioaddr, &page_size, nullptr);
  if (res == FPGA_OK) {
    printf("Page size is 2MB\n");
    return 2;
  }

  res =
      mpfVtpPrepareBuffer(MPFHandle_, span, &host_addr, FPGA_BUF_PREALLOCATED);
  if (res != FPGA_OK) {
    printf("Error mpfVtpPrepareBuffer\n");
    return -1;
  }

  pinned_mem[host_addr] = span;
  printf("Pinned memory \n");
  return 0;
}

void read_reg(fpga_handle mmio_handle, uint64_t offset, const char *name) {
  fpga_result res = FPGA_OK;
  uint64_t regval = 0;
  res = fpgaReadMMIO64(mmio_handle, 0, 0x20080 + offset, &regval);
  if (res != FPGA_OK) {
    printf("Error reading %s\n", name);
  } else {
    printf("DMA status: %s\t 0x%lx\n", name, regval);
  }
}

void read_status_reg(fpga_handle mmio_handle) {
  read_reg(mmio_handle, 0x3 * 0x8, "cmdq");
  read_reg(mmio_handle, 0x4 * 0x8, "data");
  read_reg(mmio_handle, 0x5 * 0x8, "config");
  read_reg(mmio_handle, 0x6 * 0x8, "status");
  read_reg(mmio_handle, 0x7 * 0x8, "burst_cnt");
  read_reg(mmio_handle, 0x8 * 0x8, "read_valid_cnt");
  read_reg(mmio_handle, 0x9 * 0x8, "magic_num_cnt");
  read_reg(mmio_handle, 0xA * 0x8, "wrdata_cnt");
  read_reg(mmio_handle, 0xB * 0x8, "status2");
}

// Setting the DMA CSR values
void set_dma_descriptor(fpga_handle mmio_handle, uint64_t ch_address,
                        uint64_t src_addr, uint64_t dst_addr, uint64_t len) {

  fpga_result res = FPGA_OK;

  res = fpgaWriteMMIO64(mmio_handle, 0, ch_address + 0x0, src_addr);
  if (res != FPGA_OK)
    printf("error src addr\n");
  res = fpgaWriteMMIO64(mmio_handle, 0, ch_address + 0x8, dst_addr);
  if (res != FPGA_OK)
    printf("error dst addr\n");
  res = fpgaWriteMMIO64(mmio_handle, 0, ch_address + 0x10, len);
  if (res != FPGA_OK)
    printf("error length\n");
}

void AlteraOpaeExternalRegion::reading_dma(void *Destination, size_t Source,
                                           size_t Bytes) {

  fpga_result res = FPGA_OK;
  volatile uint64_t *fpga_write_addr;
  uint64_t read_host_addr = 0;
  int ret = 0;

  // Write fence implementation
  const uint64_t wait_fpga_write_csr = 0x30;
  res = mpfVtpPrepareBuffer(
      MPFHandle_, 4 * KB,
      reinterpret_cast<void **>(const_cast<uint64_t **>(&fpga_write_addr)), 0);
  if (res != FPGA_OK)
    printf("Error allocating write_fence buffer\n");
  res = fpgaWriteMMIO64(
      MMIOHandle_, 0, 0x20000 + wait_fpga_write_csr,
      reinterpret_cast<uint64_t>(const_cast<uint64_t *>(fpga_write_addr)));
  if (res != FPGA_OK)
    printf("Error fpgaWriteMMIO64\n");

  res = mpfVtpPrepareBuffer(MPFHandle_, dma_buffer_sz, &dma_buffer_read, 0);
  if (res != FPGA_OK)
    printf("Error allocating DMA buffer\n");

  if (Bytes > buf_threshold) {
    printf("Buffer size is greater than 1MB\n");
    ret = pinning_memory(Destination, Bytes);
    if (ret < 0)
      printf("Error in pinning memory\n");
    read_host_addr = reinterpret_cast<uint64_t>(Destination);
  } else {
    read_host_addr = reinterpret_cast<uint64_t>(dma_buffer_read);
  }
  assert(read_host_addr != 0);
  printf("Host address: 0x%lX\n", read_host_addr);

  set_dma_descriptor(MMIOHandle_, 0x20100, Source, read_host_addr, Bytes);

  const uint64_t FPGA_DMA_WF_MAGIC_NO = 0x5772745F53796E63ULL;
  while (*fpga_write_addr != FPGA_DMA_WF_MAGIC_NO)
    ;
  *fpga_write_addr = 0;

  if (Bytes <= buf_threshold)
    memcpy(Destination, dma_buffer_read, Bytes);

  if (Bytes > buf_threshold && ret == 0) {
    const uint64_t page_sz = 1 << 12;
    const uint64_t page_mask = ~((page_sz)-1);
    uint64_t start_page = read_host_addr & page_mask;
    void *addr = reinterpret_cast<void *>(start_page);
    fpga_result res = mpfVtpReleaseBuffer(MPFHandle_, addr);

    if (res != FPGA_OK) {
      fprintf(stderr, "Error mpfVtpReleaseBuffer %s\n", fpgaErrStr(res));
    }
  }
}

void AlteraOpaeExternalRegion::writing_dma(size_t Destination, void *Source,
                                           size_t Bytes) {

  fpga_result res = FPGA_OK;
  fpga_event_handle event_handle;
  pollfd int_event_fd{0};
  uint64_t write_host_addr = 0;
  int ret = 0;

  // Create interrupt event
  res = fpgaCreateEventHandle(&event_handle);
  if (res != FPGA_OK)
    printf("error fpgaCreateEventHandle\n");
  res = fpgaRegisterEvent(MMIOHandle_, FPGA_EVENT_INTERRUPT, event_handle, 0);
  if (res != FPGA_OK)
    printf("error fpgaRegisterEvent\n");
  res = fpgaGetOSObjectFromEventHandle(event_handle, &int_event_fd.fd);
  if (res != FPGA_OK)
    printf("error fpgaGetOSObjectFromEventHandle\n");

  if (Bytes > buf_threshold) {
    printf("Buffer size is greater than 1MB\n");
    ret = pinning_memory(Source, Bytes);
    if (ret < 0)
      printf("Error in pinning memory\n");
    write_host_addr = reinterpret_cast<uint64_t>(Source);
  } else {
    write_host_addr = reinterpret_cast<uint64_t>(dma_buffer_write);
    memcpy(dma_buffer_write, Source, Bytes);
  }
  assert(write_host_addr != 0);
  printf("Host address: 0x%lX\n", write_host_addr);

  set_dma_descriptor(MMIOHandle_, 0x20080, write_host_addr, Destination, Bytes);

  int_event_fd.events = POLLIN;
  int poll_res = poll(&int_event_fd, 1, TIMEOUT);
  printf("Poll response: %d\n", poll_res);
  if (poll_res < 0) {
    printf("Poll error\n");
  } else if (poll_res == 0) {
    printf("Poll timeout\n");
    read_status_reg(MMIOHandle_);
    {
      fprintf(stderr, "Print some mpf stats\n");

      mpf_vtp_stats vtp_stats;
      mpfVtpGetStats(MPFHandle_, &vtp_stats);

      printf("#   VTP failed:            %ld\n",
             vtp_stats.numFailedTranslations);
      if (vtp_stats.numFailedTranslations) {
        printf("#   VTP failed addr:       0x%lx\n",
               (uint64_t)vtp_stats.ptWalkLastVAddr);
      }
      printf("#   VTP PT walk cycles:    %ld\n", vtp_stats.numPTWalkBusyCycles);
      printf("#   VTP L2 4KB hit / miss: %ld / %ld\n", vtp_stats.numTLBHits4KB,
             vtp_stats.numTLBMisses4KB);
      printf("#   VTP L2 2MB hit / miss: %ld / %ld\n", vtp_stats.numTLBHits2MB,
             vtp_stats.numTLBMisses2MB);
    }
    printf("\n");
  } else {
    uint64_t count;
    ssize_t bytes_read = read(int_event_fd.fd, &count, sizeof(count));
    printf("Bytes read: %zu\n", bytes_read);
    if (bytes_read < 0) {
      fprintf(stderr, "Error: poll failed %s\n", strerror(errno));
    }
    if (bytes_read == 0) {
      fprintf(stderr, "Error: poll failed zero bytes read\n");
    }
  }
}

AlteraOpaeExternalRegion::AlteraOpaeExternalRegion(size_t Address,
                                                   size_t RegionSize,
                                                   void *Device) {

  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Initializing AlteraOpaeExternalRegion with Address %zu "
      "and Size %zu and device %p\n",
      Address, RegionSize, Device);
  PhysAddress_ = Address;
  Size_ = RegionSize;
  MMIOHandle_ = Device;

  uint64_t mpf_mmio_offset = SVM_CCIP_MMD_MPF;

  fpga_result res = mpfConnect(MMIOHandle_, 0, mpf_mmio_offset, &MPFHandle_, 0);
  if (res != FPGA_OK) {
    printf("Error mpfConnect\n");
  }

  res = mpfVtpPrepareBuffer(MPFHandle_, dma_buffer_sz, &dma_buffer_read, 0);
  if (res != FPGA_OK)
    printf("Error allocating DMA read buffer\n");

  res = mpfVtpPrepareBuffer(MPFHandle_, dma_buffer_sz, &dma_buffer_write, 0);
  if (res != FPGA_OK)
    printf("Error allocating DMA write buffer\n");

  pocl_init_mem_region(&AllocRegion_, 0, RegionSize);
}
void AlteraOpaeExternalRegion::setDeviceHandle(fpga_handle handle) {
  MMIOHandle_ = handle;
}

AlteraOpaeExternalRegion::~AlteraOpaeExternalRegion() {

  if (MPFHandle_) {
    fpga_result res = mpfDisconnect(MPFHandle_);
    if (res != FPGA_OK)
      printf("Error mpfDisconnect\n");
  }
}

void AlteraOpaeExternalRegion::freeBuffer(pocl_mem_identifier *P) {
  pocl_free_chunk((chunk_info_t *)P->mem_ptr);
  P->mem_ptr = NULL;
}

uint64_t AlteraOpaeExternalRegion::pointerDeviceOffset(pocl_mem_identifier *P) {
  assert(P->mem_ptr);
  return ((chunk_info_t *)P->mem_ptr)->start_address;
}

// Buffer allocation uses XRT buffer allocation API
cl_int AlteraOpaeExternalRegion::allocateBuffer(pocl_mem_identifier *P,
                                                size_t Size) {
  chunk_info_t *chunk = NULL;
  chunk = pocl_alloc_buffer_from_region(&AllocRegion_, Size);
  if (chunk == NULL) {
    POCL_MSG_WARN("Almaif OPAE: Can't allocate %lu bytes for buffer\n", Size);
    return CL_FAILED;
  }
  P->mem_ptr = chunk;

  uint64_t PhysAddress = pointerDeviceOffset(P);
  POCL_MSG_PRINT_ALMAIF(
      "XRTMMAP: Initialized AlteraOpaeExternalRegion buffer with "
      "physical address %" PRIu64 "\n",
      PhysAddress);
  return CL_SUCCESS;
}

void AlteraOpaeExternalRegion::CopyToMMAP(pocl_mem_identifier *DstMemId,
                                          void *Source, size_t Bytes,
                                          size_t Offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Writing 0x%zx bytes to buffer at 0x%zx with "
      "address 0x%zx\n",
      Bytes, PhysAddress_, pointerDeviceOffset(DstMemId));
  auto src = (uint32_t *)Source;
  assert(Offset < Size_ && "Attempt to access data outside XRT memory");

  if (!pocl_get_bool_option("POCL_OPAE_SIM", false)) {
    writing_dma(pointerDeviceOffset(DstMemId) + Offset, Source, Bytes);
  }
}

void AlteraOpaeExternalRegion::CopyFromMMAP(void *Destination,
                                            pocl_mem_identifier *SrcMemId,
                                            size_t Bytes, size_t Offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Reading 0x%zx bytes from buffer at 0x%zx "
      "with address 0x%zx\n",
      Bytes, PhysAddress_, pointerDeviceOffset(SrcMemId));
  assert(Offset < Size_ && "Attempt to access data outside XRT memory");

  if (!pocl_get_bool_option("POCL_OPAE_SIM", false)) {
    reading_dma(Destination, pointerDeviceOffset(SrcMemId) + Offset, Bytes);
  }
}

void AlteraOpaeExternalRegion::CopyInMem(size_t Source, size_t Destination,
                                         size_t Bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Copying 0x%zx bytes from 0x%zx "
                             "to 0x%zx\n",
                             Bytes, Source, Destination);
  size_t SrcOffset = Source - PhysAddress_;
  size_t DstOffset = Destination - PhysAddress_;
  assert(SrcOffset < Size_ && (SrcOffset + Bytes) <= Size_ &&
         "Attempt to access data outside XRT memory");
  assert(DstOffset < Size_ && (DstOffset + Bytes) <= Size_ &&
         "Attempt to access data outside XRT memory");
  //  assert(DeviceBuffer != XRT_NULL_HANDLE &&
  //         "No kernel handle; write before mapping?");
  /*
    xrt::bo *b = (xrt::bo *)DeviceBuffer;
    auto b_mapped = b->map();

    b->sync(XCL_BO_SYNC_BO_FROM_DEVICE, Bytes, SrcOffset);
    memcpy((char *)b_mapped + DstOffset, (char *)b_mapped + SrcOffset, Bytes);
    b->sync(XCL_BO_SYNC_BO_TO_DEVICE, Bytes, DstOffset);
  */
}
