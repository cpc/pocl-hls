/* AlteraOpaeRegion.cc - Access on-chip memory of an OPAE device as AlmaIFRegion

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
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <unistd.h>

#include <opae/fpga.h>
#include <opae/mpf/mpf.h>

#include "aocl_mmd.h"
#include "ccip_mmd.h"
#include "ccip_mmd_device.h"
#include "fpgaconf.h"

#include "AlteraOpaeRegion.hh"

#define KERNEL_SYSTEM_OFFSET 0x1000

#define SPAN_WIDTH 12
#define SPAN_CTRL_OFFSET 0x20
static size_t CurrentSpan_ = 0;

// Span extender logic
size_t AlteraOpaeRegion::setSpan(size_t Offset) {

  size_t OffsetMask = (1 << SPAN_WIDTH) - 1;
  size_t SpanMask = ~OffsetMask;

  size_t newSpan = Offset & SpanMask;

  // if (newSpan != CurrentSpan_) {
  fpga_result res =
      fpgaWriteMMIO64(MMIOHandle_, 0, 0x4000 + SPAN_CTRL_OFFSET, newSpan);
  if (res != FPGA_OK)
    printf("Error writing span\n");
  CurrentSpan_ = newSpan;
  // }

  size_t maskedOffset = Offset & OffsetMask;

  POCL_MSG_PRINT_ALMAIF_MMAP("OPAE: Span:%zx Offset in span:%zx\n",
                             CurrentSpan_, maskedOffset);
  return maskedOffset;
}

AlteraOpaeRegion::AlteraOpaeRegion(size_t Address, size_t RegionSize,
                                   void *kernel, size_t DeviceOffset) {

  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Initializing AlteraOpaeRegion with Address %zu "
      "and Size %zu and kernel %p and DeviceOffset 0x%zx\n",
      Address, RegionSize, kernel, DeviceOffset);
  PhysAddress_ = Address;
  Size_ = RegionSize;
  MMIOHandle_ = kernel;
  DeviceOffset_ = DeviceOffset;
}

AlteraOpaeRegion::AlteraOpaeRegion(size_t Address, size_t RegionSize,
                                   void *kernel, const std::string &init_file,
                                   size_t DeviceOffset)
    : AlteraOpaeRegion(Address, RegionSize, kernel, DeviceOffset) {

  if (RegionSize == 0) {
    return; // don't try to write to empty region
  }
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Initializing AlteraOpaeRegion with file %s\n",
      init_file.c_str());
  initRegion(init_file);
}

uint32_t AlteraOpaeRegion::Read32(size_t offset) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Reading from region at 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_,
                             PhysAddress_ + offset - DeviceOffset_);
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");
  uint32_t value = 0;

  size_t offset_in_span = setSpan(PhysAddress_ + offset - DeviceOffset_);
  // read(KERNEL_SYSTEM_OFFSET + offset_in_span, &value, 4);
  fpga_result res = FPGA_OK;
  res = fpgaReadMMIO32(MMIOHandle_, 0,
                       0x4000 + KERNEL_SYSTEM_OFFSET + offset_in_span, &value);
  if (res != FPGA_OK)
    printf("Error reading\n");

  return value;
}

void AlteraOpaeRegion::Write32(size_t offset, uint32_t value) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Writing to region at 0x%zx with "
                             "offset 0x%zx\n",
                             PhysAddress_,
                             PhysAddress_ + offset - DeviceOffset_);
  assert(offset < Size_ && "Attempt to access data outside MMAP'd buffer");

  size_t offset_in_span = setSpan(PhysAddress_ + offset - DeviceOffset_);
  fpga_result res = fpgaWriteMMIO32(
      MMIOHandle_, 0, 0x4000 + KERNEL_SYSTEM_OFFSET + offset_in_span, value);
  if (res != FPGA_OK)
    printf("Error writing\n");
}

void AlteraOpaeRegion::CopyToMMAP(size_t destination, const void *source,
                                  size_t bytes) {
  auto src = (uint32_t *)source;
  size_t offset = destination - PhysAddress_;
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "OPAEMMAP: Writing 0x%zx bytes to buffer at region 0x%zx with "
      "address 0x%zx and offset %zx\n",
      bytes, PhysAddress_, destination, offset);
  assert(offset < Size_ && "Attempt to access data outside XRT memory");

  assert((offset & 0b11) == 0 &&
         "Xrt copytommap destination must be 4 byte aligned");
  assert(((size_t)src & 0b11) == 0 &&
         "Xrt copytommap source must be 4 byte aligned");
  assert((bytes % 4) == 0 && "Xrt copytommap size must be 4 byte multiple");

  for (size_t i = 0; i < bytes / 4; ++i) {
    Write32(destination + 4 * i - PhysAddress_, src[i]);
  }
}

void AlteraOpaeRegion::CopyFromMMAP(void *destination, size_t source,
                                    size_t bytes) {
  auto dst = (uint32_t *)destination;
  size_t offset = source - PhysAddress_;
  POCL_MSG_PRINT_ALMAIF_MMAP(
      "XRTMMAP: Reading 0x%zx bytes from region at 0x%zx "
      "with address 0x%zx and offset 0x%zx\n",
      bytes, PhysAddress_, source, offset);
  assert(offset < Size_ && "Attempt to access data outside XRT memory");
  assert((offset & 0b11) == 0 &&
         "Xrt copyfrommmap source must be 4 byte aligned");

  switch (bytes) {
  case 1: {
    uint32_t value = Read32(source - DeviceOffset_);
    *((uint8_t *)destination) = value;
    break;
  }
  case 2: {
    uint32_t value = Read32(source - DeviceOffset_);
    *((uint16_t *)destination) = value;
    break;
  }
  default: {
    assert(((size_t)dst & 0b11) == 0 &&
           "Xrt copyfrommmap destination must be 4 byte aligned");
    size_t i;
    for (i = 0; i < bytes / 4; ++i) {
      dst[i] = Read32(source - DeviceOffset_ + 4 * i);
    }
    if ((bytes % 4) != 0) {
      union value {
        char bytes[4];
        uint32_t full;
      } value1;
      value1.full = Read32(source - DeviceOffset_ + 4 * i);
      for (int k = 0; k < (bytes % 4); k++) {
        dst[i] = value1.bytes[k];
      }
    }
  }
  }
}

void AlteraOpaeRegion::CopyInMem(size_t source, size_t destination,
                                 size_t bytes) {
  POCL_MSG_PRINT_ALMAIF_MMAP("XRTMMAP: Copying 0x%zx bytes from 0x%zx "
                             "to 0x%zx\n",
                             bytes, source, destination);
  size_t src_offset = source - PhysAddress_;
  size_t dst_offset = destination - PhysAddress_;
  assert(src_offset < Size_ && (src_offset + bytes) <= Size_ &&
         "Attempt to access data outside XRT memory");
  assert(dst_offset < Size_ && (dst_offset + bytes) <= Size_ &&
         "Attempt to access data outside XRT memory");
  assert((bytes % 4) == 0 && "Xrt copyinmem size must be 4 byte multiple");

  for (size_t i = 0; i < bytes / 4; ++i) {
    uint32_t m = Read32(source - DeviceOffset_ + 4 * i);
    Write32(destination - DeviceOffset_ + 4 * i, m);
  }
}

void AlteraOpaeRegion::setKernelPtr(void *ptr) { MMIOHandle_ = ptr; }
