/* firmware.c - Firmware for openasip device implementing AlmaIF, communicating
 * with MLIR HLS-generated accelerator

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

#include <stdint.h>
#include <string.h>

#include "pocl_context.h"

#ifndef QUEUE_LENGTH
#define QUEUE_LENGTH 3
#endif

#define AQL_PACKET_INVALID (1)
#define AQL_PACKET_KERNEL_DISPATCH (2)
#define AQL_PACKET_BARRIER_AND (3)
#define AQL_PACKET_AGENT_DISPATCH (4)
#define AQL_PACKET_BARRIER_OR (5)
#define AQL_PACKET_BARRIER (1 << 8)
#define AQL_PACKET_LENGTH (64)

#define AQL_MAX_SIGNAL_COUNT (5)

#define ALMAIF_STATUS_REG (0x00)
#define ALMAIF_STATUS_REG_PC (0x04)
#define ALMAIF_STATUS_REG_CC_LOW (0x08)
#define ALMAIF_STATUS_REG_CC_HIGH (0x0C)
#define ALMAIF_STATUS_REG_SC_LOW (0x10)
#define ALMAIF_STATUS_REG_SC_HIGH (0x14)

#define SLEEP_CYCLES 400

#ifndef QUEUE_START
#define QUEUE_START 0
#endif

#include "almaif-tce-device-defs.h"
#include "kernel_arg_setter.h"

#define MM2S_PTR_OFFSET (0x8 / 4)
#define MM2S_LEN_OFFSET (0x18 / 4)
#define S2MM_PTR_OFFSET (0x10 / 4)
#define S2MM_LEN_OFFSET (0x18 / 4)

void
launch_workgroup_function (__cq__ volatile struct AQLDispatchPacket *packet);

int
main ()
{
  __cq__ volatile struct AQLQueueInfo *queue_info
    = (__cq__ volatile struct AQLQueueInfo *)QUEUE_START;
  int read_iter = queue_info->read_index_low;

  queue_info->base_address_high = 42;
  uint8_t run_commands[16];
  memset (run_commands, 0b1, 16);
  while (1)
    {
      // Compute packet location
      uint32_t packet_loc = QUEUE_START + AQL_PACKET_LENGTH
                            + ((read_iter % QUEUE_LENGTH) * AQL_PACKET_LENGTH);
      __cq__ volatile struct AQLDispatchPacket *packet
        = (__cq__ volatile struct AQLDispatchPacket *)packet_loc;
      // The driver will mark the packet as not INVALID when it wants us to
      // compute it
      //
      // queue_info->doorbell_signal_low = 47;
      while (packet->header == AQL_PACKET_INVALID)
        ;
      uint16_t header = packet->header;
      queue_info->type = header;

      __global__ struct CommandMetadata *cmd_meta
        = (__global__ struct CommandMetadata *)packet->cmd_metadata_low;
      if (header & (1 << AQL_PACKET_BARRIER_AND))
        {
          queue_info->doorbell_signal_low = 152;
          queue_info->type = header;
          __cq__ volatile struct AQLAndPacket *andPacket
            = (__cq__ volatile struct AQLAndPacket *)packet_loc;

          for (int i = 0; i < AQL_MAX_SIGNAL_COUNT; i++)
            {
              volatile __global__ uint32_t *signal
                = (volatile __global__ uint32_t *)(andPacket
                                                     ->dep_signals[2 * i]);
              if (signal != 0)
                {
                  while (*signal == 0)
                    {
                      for (int kk = 0; kk < SLEEP_CYCLES; kk++)
                        {
                          asm volatile ("...;");
                        }
                    }
                }
            }
        }
      else if (header & (1 << AQL_PACKET_KERNEL_DISPATCH))
        {
          const int work_dim = packet->dimensions;
          struct pocl_context32 __global__ *context
            = (struct pocl_context32 __global__ *)(packet
                                                     ->pocl_context32b_low);

          const int num_groups_x = context->num_groups[0];
          const int num_groups_y
            = (work_dim >= 2) ? (context->num_groups[1]) : 1;
          const int num_groups_z
            = (work_dim == 3) ? (context->num_groups[2]) : 1;

          const unsigned lsize_x = context->local_size[0];
          const int lsize_y = (work_dim >= 2) ? (context->local_size[1]) : 1;
          const int lsize_z = (work_dim == 3) ? (context->local_size[2]) : 1;

          __global__ uint32_t *kernarg_ptr
            = (__global__ uint32_t *)(packet->kernarg_address_low);

          unsigned kernel_id = cmd_meta->kernel_func_ptr_low;
          __global__ volatile uint32_t *ACCELERATOR
            = (__global__ volatile uint32_t *)(0x10000 * kernel_id);

          for (unsigned gid_x = 0; gid_x < num_groups_x; gid_x++)
            for (unsigned gid_y = 0; gid_y < num_groups_y; gid_y++)
              for (unsigned gid_z = 0; gid_z < num_groups_z; gid_z++)
                {
                  almaif_mlir_arg_setter (kernarg_ptr, kernel_id, context,
                                          gid_x, gid_y, gid_z);

                  ACCELERATOR[0] = run_commands[kernel_id];
                  run_commands[kernel_id] = 0b10001;
                  uint32_t status = 0;
                  while (status != 0b1010)
                    {
                      status = ACCELERATOR[0] & 0b1010;
                    }
                }
        }
      // Completion signal is given as absolute address
      if (cmd_meta)
        {
          cmd_meta->completion_signal = 1;
        }
      packet->header = AQL_PACKET_INVALID;

      read_iter++; // move on to the next AQL packet
      queue_info->read_index_low = read_iter;
    }
}
