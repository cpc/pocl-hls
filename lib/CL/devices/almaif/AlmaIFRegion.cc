/* AlmaIFRegion.cc - Interface class for raw memory operations
 * (read and write operations to backend-specific memory)

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

#include "AlmaIFRegion.hh"

#include "pocl_util.h"

#include <fstream>

AlmaIFRegion::~AlmaIFRegion() {}

bool AlmaIFRegion::isInRange(size_t dst) {
  return ((dst >= PhysAddress_) && (dst < (PhysAddress_ + Size_)));
}

size_t AlmaIFRegion::PhysAddress() { return PhysAddress_; }

size_t AlmaIFRegion::Size() { return Size_; }

void AlmaIFRegion::Write32(size_t offset, uint32_t value) {
  CopyToMMAP(PhysAddress() + offset, &value, 4);
}

uint32_t AlmaIFRegion::Read32(size_t offset) {
  uint32_t value = 0;
  CopyFromMMAP(&value, PhysAddress() + offset, 4);
  return value;
}

void AlmaIFRegion::Write64(size_t offset, uint64_t value) {
  Write32(offset, (uint32_t)value);
  Write32(offset + 4, (uint32_t)(value >> 32));
}

uint64_t AlmaIFRegion::Read64(size_t offset) {
  uint32_t low_bits = Read32(offset);
  uint32_t high_bits = Read32(offset + 4);
  uint64_t value = ((uint64_t)high_bits << 32) | low_bits;
  return value;
}

void AlmaIFRegion::Write16(size_t offset, uint16_t value) {
  uint32_t old_value = Read32(offset & 0xFFFFFFFC);
  uint32_t new_value = 0;
  if ((offset & 0b10) == 0) {
    new_value = (old_value & 0xFFFF0000) | (uint32_t)value;
  } else {
    new_value = ((uint32_t)value << 16) | (old_value & 0xFFFF);
  }
  Write32(offset & 0xFFFFFFFC, new_value);
}

void AlmaIFRegion::initRegion(const std::string &init_file) {
  std::ifstream inFile;
  inFile.open(init_file.c_str(), std::ios::binary);
  unsigned int current;
  int i = 0;
  while (inFile.good()) {
    inFile.read(reinterpret_cast<char *>(&current), sizeof(current));
    Write32(i, current);
    i += 4;
  }

  POCL_MSG_PRINT_ALMAIF_MMAP("MMAP: Initialized region with %i bytes \n",
                             i - 4);
}
