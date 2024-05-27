/* AlteraOpaeExternalRegion.hh - Access external memory (DDR or HBM) of an OPAE
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

#ifndef POCL_ALTERAOPAEEXTERNALREGION_H
#define POCL_ALTERAOPAEEXTERNALREGION_H

#include <stdlib.h>

#include "bufalloc.h"
#include <opae/fpga.h>
#include <opae/mpf/mpf.h>

class AlteraOpaeExternalRegion {
public:
  AlteraOpaeExternalRegion(size_t Address, size_t RegionSize, void *Device);
  ~AlteraOpaeExternalRegion();

  void CopyToMMAP(pocl_mem_identifier *DstMemId, void *Source, size_t Bytes,
                  size_t Offset);
  void CopyFromMMAP(void *Destination, pocl_mem_identifier *SrcMemId,
                    size_t Bytes, size_t Offset);
  void CopyInMem(size_t Source, size_t Destination, size_t Bytes);

  void setDeviceHandle(void *handle);

  // Returns the offset of the allocated pointer in the DDR address space
  // used by the kernel
  uint64_t pointerDeviceOffset(pocl_mem_identifier *P);
  cl_int allocateBuffer(pocl_mem_identifier *P, size_t Size);
  void freeBuffer(pocl_mem_identifier *P);

private:
  int pinning_memory(void *dma_host_addr, size_t len);
  void writing_dma(size_t Destination, void *Source, size_t Bytes);
  void reading_dma(void *Destination, size_t Source, size_t Bytes);
  size_t Size_;
  size_t PhysAddress_;
  fpga_handle MMIOHandle_;
  mpf_handle_t MPFHandle_;
  memory_region_t AllocRegion_;
};

#endif
