/* AlmaifCompileMLIR.cc - compiler support for HLS Almaif device

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

#include "CL/cl_ext.h"
#include "config.h"
#include "config2.h"

#include "common.h"
#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_debug.h"
#include "pocl_file_util.h"
#include "pocl_llvm.h"
#include "pocl_mlir.h"

#include "stdint.h"
#include "unistd.h"

#include <iostream>
#include <sstream>
#include <string>

#include "../AlmaifCompile.hh"
#include "../AlmaifShared.hh"
#include "AlmaifCompileMLIR.hh"

extern int pocl_offline_compile;

static const char *final_emulation_ld_flags[] = {"-lm", "-nostartfiles",
                                                 HOST_LD_FLAGS_ARRAY, NULL};

int pocl_almaif_mlir_initialize(cl_device_id device,
                                const std::string &parameters) {
  AlmaifData *d = (AlmaifData *)(device->data);

  mlir_backend_data_t *bd = new mlir_backend_data_t();
  if (bd == NULL) {
    POCL_MSG_WARN("couldn't allocate mlir_backend_data\n");
    return CL_OUT_OF_HOST_MEMORY;
  }
  POCL_INIT_LOCK(bd->mlir_compile_lock);

  device->device_side_printf = 0;
  device->endian_little = 1;
  bd->core_count = 1;

  device->long_name = device->short_name = "ALMAIF MLIR";
  device->vendor = "pocl";
  device->kernellib_name = "kernel-mlirbc";
  device->kernellib_fallback_name = NULL;
  device->kernellib_subdir = "mlir";
  if (!pocl_offline_compile && d->Dev->isEmulationDevice()) {
    device->llvm_cpu = pocl_get_llvm_cpu_name();
    device->llvm_target_triplet = OCL_KERNEL_TARGET;
    device->llvm_abi = pocl_get_llvm_cpu_abi();
    device->final_linkage_flags = final_emulation_ld_flags;
    POCL_MSG_PRINT_ALMAIF("Initializing Emulation device for compilation\n");
  }
  device->type = CL_DEVICE_TYPE_ACCELERATOR;
  d->compilationData->backend_data = (void *)bd;
  return 0;
}

int pocl_almaif_mlir_cleanup(cl_device_id device) {
  void *data = device->data;
  AlmaifData *d = (AlmaifData *)data;
  mlir_backend_data_t *bd =
      (mlir_backend_data_t *)d->compilationData->backend_data;
  POCL_DESTROY_LOCK(bd->mlir_compile_lock);
  delete bd;
  return 0;
}

void pocl_almaif_mlir_generate_hlscpp_from_hlsmlir(
    const std::string &kernel_mlir_path, const std::string &kernel_hlscpp_path,
    bool emitOpenCL) {

  std::string invokeMlir = std::string(SCALEHLS_TRANSLATE_EXECUTABLE) + " -o ";
  invokeMlir += kernel_hlscpp_path;
  if (emitOpenCL) {
    invokeMlir += " --scalehls-emit-opencl ";
  } else {
    invokeMlir += " --scalehls-emit-hlscpp ";
    invokeMlir += " --emit-vitis-directives ";
  }
  invokeMlir += kernel_mlir_path;
  POCL_MSG_PRINT_ALMAIF("MLIR-Translate cmd: %s\n", invokeMlir.c_str());
  system(invokeMlir.c_str());
  if (emitOpenCL) {
    POCL_ABORT("INTEL FLOW UNIMPLEMENTED\n");
  }
}

void pocl_almaif_mlir_generate_vitis_zip_from_hlscpp(
    const std::string &cachedir, const char *kernel_name) {
  // Changing into the cachedir is necessary because vitis HLS can only create
  // projects to the current working directory
  std::string invokeVitis = "SAVEDIR=$PWD; cd " + cachedir;
  invokeVitis += "; vitis_hls -f ";
  invokeVitis += SRCDIR;
  invokeVitis += "/lib/CL/devices/almaif/mlir/generate_hls_core.tcl";

  invokeVitis += " -tclargs pocl_mlir_";
  invokeVitis += kernel_name;

  invokeVitis += " -tclargs .";
  invokeVitis += POCL_PARALLEL_HLSCPP_FILENAME;

  invokeVitis += " -tclargs ";
  invokeVitis += "kernel_" + std::string(kernel_name) + "_vitis.zip";

  // Only relative path needed here as we have cd:d to the cachedir
  invokeVitis += " -tclargs vitis_project_" + std::string(kernel_name);

  invokeVitis += "; cd $SAVEDIR";

  POCL_MSG_PRINT_ALMAIF("Vitis cmd: %s\n", invokeVitis.c_str());
  system(invokeVitis.c_str());
}

void pocl_almaif_mlir_generate_xo_from_rtl(
    const std::vector<std::string> &kernel_vitis_zip_paths,
    const std::string &kernel_xo_path,
    const std::string &kernel_xo_project_path,
    const std::vector<std::string> &kernel_names,
    const std::vector<std::string> &workgroup_function_names,
    const std::vector<std::string> &kernel_vitis_folder_paths,
    const std::vector<std::string> &kernel_iprepo_paths) {
  pocl_rm_rf(kernel_xo_project_path.c_str());

  std::string invokeVivado = "vivado -mode batch -source ";
  invokeVivado += SRCDIR;
  invokeVivado += "/lib/CL/devices/almaif/mlir/generate_xo.tcl";

  invokeVivado += " -tclargs ";
  invokeVivado += kernel_xo_project_path;

  invokeVivado += " -tclargs ";
  invokeVivado += kernel_xo_path;

  invokeVivado += " -tclargs ";
  invokeVivado += std::to_string(kernel_names.size());

  invokeVivado += " -tclargs ";
  invokeVivado += BUILDDIR "/lib/CL/devices/almaif/rtl_tta_core";

  for (int i = 0; i < kernel_names.size(); i++) {
    invokeVivado += " -tclargs ";
    invokeVivado += kernel_names[i];

    invokeVivado += " -tclargs ";
    invokeVivado += workgroup_function_names[i];

    invokeVivado += " -tclargs ";
    invokeVivado += kernel_vitis_folder_paths[i];

    invokeVivado += " -tclargs ";
    invokeVivado += kernel_iprepo_paths[i];
  }

  POCL_MSG_PRINT_ALMAIF("Vivado cmd: %s\n", invokeVivado.c_str());
  system(invokeVivado.c_str());

  assert(pocl_exists(kernel_xo_path.c_str()));
}

void pocl_almaif_mlir_generate_xclbin_from_xo(
    const std::string &kernel_xo_path, const std::string &kernel_xclbin_path,
    const std::string &cachedir,
    const std::vector<std::string> &kernelVitisFolderPaths,
    const std::vector<std::string> &kernelIpRepoPaths) {

  std::string emulationOrHWTarget =
      pocl_get_string_option("XCL_EMULATION_MODE", "hw");

  std::string invokeVpp = "v++ -t ";
  invokeVpp += emulationOrHWTarget;
  invokeVpp += " -l --platform ";
  invokeVpp += "/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/"
               "xilinx_u280_gen3x16_xdma_1_202211_1.xpfm";

  invokeVpp += " --temp_dir ./_x.";
  invokeVpp += emulationOrHWTarget;

  if (emulationOrHWTarget == "hw") {
    invokeVpp += " -s -g -R 2";
    invokeVpp += " --remote_ip_cache vpp_ip_cache";
  } else {
    invokeVpp += " -R 1";
    invokeVpp += " --vivado.prop run.xsim.simulate.log_all_signals=false";
  }

  for (const auto &kernel_vitis_folder_path : kernelVitisFolderPaths) {
    invokeVpp += " --user_ip_repo_paths ";
    invokeVpp += kernel_vitis_folder_path;
  }
  invokeVpp += " --user_ip_repo_paths ";
  invokeVpp += kernelIpRepoPaths[0];

  invokeVpp += " " + kernel_xo_path;

  invokeVpp += " -o ";
  invokeVpp += kernel_xclbin_path;

  POCL_MSG_PRINT_ALMAIF("V++ cmd: %s\n", invokeVpp.c_str());
  system(invokeVpp.c_str());

  if (!pocl_exists(kernel_xclbin_path.c_str())) {
    POCL_ABORT("Failed generating initial version of %s\n",
               kernel_xclbin_path.c_str());
  }

  std::string embeddedMetadatafile = cachedir;
  embeddedMetadatafile += "/embeddedMetadata.xml";

  std::string invokeXclbinutil =
      "xclbinutil --force --dump-section EMBEDDED_METADATA:RAW:";
  invokeXclbinutil += embeddedMetadatafile;
  invokeXclbinutil += " --input ";
  invokeXclbinutil += kernel_xclbin_path;
  POCL_MSG_PRINT_ALMAIF("Xclbinutil extract cmd: %s\n",
                        invokeXclbinutil.c_str());
  system(invokeXclbinutil.c_str());

  invokeXclbinutil = "python3 ";
  invokeXclbinutil += SRCDIR;
  invokeXclbinutil += "/lib/CL/devices/almaif/mlir/set_xclbin_range.py ";
  invokeXclbinutil += embeddedMetadatafile;
  POCL_MSG_PRINT_ALMAIF("Python3 script modifying the xclbin range cmd: %s\n",
                        invokeXclbinutil.c_str());
  system(invokeXclbinutil.c_str());

  // Temp file needed because xclbinutil cannot modify in-place
  std::string tmp_xclbin_file = cachedir;
  tmp_xclbin_file += "/tmp.xclbin";

  invokeXclbinutil =
      "xclbinutil --force --replace-section EMBEDDED_METADATA:RAW:";
  invokeXclbinutil += embeddedMetadatafile;
  invokeXclbinutil += " --input ";
  invokeXclbinutil += kernel_xclbin_path;
  invokeXclbinutil += " --output ";
  invokeXclbinutil += tmp_xclbin_file;
  POCL_MSG_PRINT_ALMAIF("Xclbinutil replace cmd: %s\n",
                        invokeXclbinutil.c_str());
  system(invokeXclbinutil.c_str());
  if (!pocl_exists(tmp_xclbin_file.c_str())) {
    POCL_ABORT("Failed generating metadata corrected version of %s\n",
               tmp_xclbin_file.c_str());
  }

  pocl_rename(tmp_xclbin_file.c_str(), kernel_xclbin_path.c_str());
}

void pocl_almaif_mlir_generate_firmware(const std::string &firmware_path,
                                        const std::string &cachedir) {
  std::string outputTpef = cachedir + "/parallel.tpef";
  std::string mainC = "firmware.c";

  std::string deviceMainSrc;
  std::string poclIncludePathSwitch;
  std::string machineFile;
  if (pocl_get_bool_option("POCL_BUILDING", 0)) {
    deviceMainSrc =
        std::string(SRCDIR) + "/lib/CL/devices/almaif/mlir/" + mainC;
    poclIncludePathSwitch = " -I " + std::string(SRCDIR) + "/include" + " -I " +
                            std::string(SRCDIR) +
                            "/lib/CL/devices/almaif/openasip";
    machineFile = std::string(SRCDIR) + "/lib/CL/devices/almaif/mlir/tta.adf";
  } else {
    deviceMainSrc = std::string(POCL_INSTALL_PRIVATE_DATADIR) + "/" + mainC;
    poclIncludePathSwitch =
        " -I " + std::string(POCL_INSTALL_PRIVATE_DATADIR) + "/include";
    machineFile = std::string(SRCDIR) + "/tta.adf";
  }
  assert(access(deviceMainSrc.c_str(), R_OK) == 0);
  assert(access(machineFile.c_str(), R_OK) == 0);

  std::string invokeOacc = "PATH=" OPENASIP_LLVM_DIR;
  invokeOacc += "/bin/:$PATH LD_LIBRARY_PATH=" OPENASIP_LLVM_DIR "/lib ";
  invokeOacc += "oacc " + poclIncludePathSwitch + " " + deviceMainSrc + " -I" +
                cachedir + " -g -O3 --little-endian -o " + outputTpef + " -a " +
                machineFile;
  POCL_MSG_PRINT_ALMAIF("Oacc cmd: %s\n", invokeOacc.c_str());
  system(invokeOacc.c_str());

  std::string invokeGeneratebits = "LD_LIBRARY_PATH=" OPENASIP_LLVM_DIR "/lib ";
  invokeGeneratebits += "generatebits --piformat=bin2n --imem-mau-pkg ";
  invokeGeneratebits += "--program " + cachedir + "/parallel.tpef ";
  invokeGeneratebits += "--output-file " + firmware_path;
  invokeGeneratebits += " " + machineFile;
  POCL_MSG_PRINT_ALMAIF("Generatebits cmd: %s\n", invokeGeneratebits.c_str());
  system(invokeGeneratebits.c_str());
}

void generateLaunchWorkgroupFunction(AlmaifData *D, int num_kernels,
                                     cl_kernel *kernels,
                                     _cl_command_node **cmds,
                                     const std::string &launcher_function_path,
                                     int specialize) {
  std::stringstream outputString;
  outputString << "void almaif_mlir_arg_setter(__global__ uint32_t* "
                  "kernarg_ptr,\n"
                  "    int accelerator_id,\n"
                  "    struct pocl_context32 __global__ *pc,\n"
                  "    unsigned gid_x, unsigned gid_y, unsigned gid_z) {\n"
                  "  switch (accelerator_id) {\n";
  for (int idx = 0; idx < num_kernels; ++idx) {
    cl_kernel kernel = kernels[idx];
    _cl_command_node *cmd = cmds[idx];
    pocl_kernel_metadata_t *meta = kernel->meta;
    assert(meta && "kernel metadata NULL");
    std::vector<unsigned> arg_buffer_idx = {};
    int current_arg_byte = 0;
    outputString << "     case " << idx << ":\n"
                 << "     {\n"
                 << "     __global__ volatile uint32_t* ACCELERATOR = "
                    "(__global__ volatile uint32_t*)(0x10000 * "
                 << idx << ");\n";

    int CUArgumentOffset = 0x10;
    for (int i = 0; i < meta->num_args; ++i) {
      struct pocl_argument *al = &(cmd->command.run.arguments[i]);
      int arg_size = meta->arg_info[i].type_size;
      if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
        // arg_size = D->Dev->PointerSize;
        // arg_size = 4;
        // outputString << "    ACCELERATOR["
        //              << "0x" << std::hex << CUArgumentOffset << std::dec
        //              << " / 4] = *(kernarg_ptr + " << (current_arg_byte / 4)
        //              << ");\n    ACCELERATOR["
        //              << "0x" << std::hex << (CUArgumentOffset + 4) <<
        //              std::dec
        //              << " / 4] = 0;\n";
        // CUArgumentOffset += 8;
        arg_buffer_idx.push_back(current_arg_byte);
        current_arg_byte += MAX_EXTENDED_ALIGNMENT;
      } else {
        switch (arg_size) {
        case 1:
          outputString
              << "    ACCELERATOR[" << "0x" << std::hex << CUArgumentOffset
              << std::dec
              << " / 4] = *((__global__ uint8_t*)((uint32_t)kernarg_ptr + "
              << current_arg_byte << "));\n";
          CUArgumentOffset += 4;
          current_arg_byte += MAX_EXTENDED_ALIGNMENT;
          break;
        case 2:
          outputString
              << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
              << std::dec
              << " / 4] = *((__global__ uint16_t*)((uint32_t)kernarg_ptr + "
              << current_arg_byte << "));\n";
          CUArgumentOffset += 4;
          current_arg_byte += MAX_EXTENDED_ALIGNMENT;
          break;
        default:
          for (int k = 0; k < arg_size / 4; k++) {
            outputString
                << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                << std::dec
                << " / 4] = *((__global__ uint32_t*)((uint32_t)kernarg_ptr + "
                << current_arg_byte << "));\n";
            CUArgumentOffset += 4;
            current_arg_byte += MAX_EXTENDED_ALIGNMENT;
          }
          break;
        }
        CUArgumentOffset += 4;
      }
    }
    bool is_cmd_buffer_kernel =
        !strncmp(kernels[0]->name, "command_buffer", 14);
    if (!is_cmd_buffer_kernel) {
      for (int i = 0; i < 3; ++i) {
        outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                     << std::dec << " / 4] = pc->num_groups[" << i << "];\n";
        CUArgumentOffset += 4;
        outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                     << std::dec << " / 4] = 0;\n";
        CUArgumentOffset += 4 + 4;
      }
      for (int i = 0; i < 3; ++i) {
        outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                     << std::dec << " / 4] = pc->global_offset[" << i << "];\n";
        CUArgumentOffset += 4;
        outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                     << std::dec << " / 4] = 0;\n";
        CUArgumentOffset += 4 + 4;
      }
      for (int i = 0; i < 3; ++i) {
        outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                     << std::dec << " / 4] = pc->local_size[" << i << "];\n";
        CUArgumentOffset += 4;
        outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                     << std::dec << " / 4] = 0;\n";
        CUArgumentOffset += 4 + 4;
      }
      outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = pc->printf_buffer;\n";
      CUArgumentOffset += 4;
      outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = 0;\n";
      CUArgumentOffset += 4 + 4;

      outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = pc->printf_buffer_position;\n";
      CUArgumentOffset += 4;
      outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = 0;\n";
      CUArgumentOffset += 4 + 4;

      outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = pc->printf_buffer_capacity;\n";
      CUArgumentOffset += 4 + 4;

      outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = pc->global_var_buffer;\n";
      CUArgumentOffset += 4;
      outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = 0;\n";
      CUArgumentOffset += 4 + 4;

      outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = pc->work_dim;\n";
      CUArgumentOffset += 4 + 4;

      outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = pc->execution_failed;\n";
      CUArgumentOffset += 4 + 4;

      for (int i = 0; i < 3; ++i) {
        char xyzChar = 'x' + i;
        outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                     << std::dec << " / 4] = gid_" << xyzChar << ";\n";
        CUArgumentOffset += 4;
        outputString << "    ACCELERATOR[0x" << std::hex << CUArgumentOffset
                     << std::dec << " / 4] = 0;\n";
        CUArgumentOffset += 4 + 4;
      }
    }
    for (auto idx : arg_buffer_idx) {
      auto arg_size = 4;
      outputString << "    ACCELERATOR[" << "0x" << std::hex << CUArgumentOffset
                   << std::dec << " / 4] = *(kernarg_ptr + " << (idx / 4)
                   << ");\n    ACCELERATOR[" << "0x" << std::hex
                   << (CUArgumentOffset + 4) << std::dec << " / 4] = 0;\n";
      CUArgumentOffset += 8 + 4;
    }
    outputString << "    } break;\n";
  }
  outputString << "  }\n"
               << "}";
  pocl_write_file(launcher_function_path.c_str(), outputString.str().c_str(),
                  outputString.str().length(), 0);
}

int pocl_almaif_mlir_compile_kernels(int num_kernels, _cl_command_node **cmds,
                                     cl_kernel *kernels, cl_device_id device,
                                     int specialize) {

  //  if (!device)
  //    device = cmd->device;
  assert(device);
  void *data = device->data;
  AlmaifData *d = (AlmaifData *)data;
  assert(d != NULL);

  mlir_backend_data_t *bd =
      (mlir_backend_data_t *)d->compilationData->backend_data;

  POCL_MSG_PRINT_ALMAIF("Starting to compile parallel.mlir's\n");
  POCL_LOCK(bd->mlir_compile_lock);

  std::vector<std::string> kernelNames;
  std::vector<std::string> kernelVitisZipPaths;
  std::vector<std::string> workgroupFunctionNames;
  std::vector<std::string> kernelVitisFolderPaths;
  std::vector<std::string> kernelIpRepoPaths;

  char xclbinDir[POCL_MAX_PATHNAME_LENGTH];
  std::string kernel_xclbin_path;
  if (pocl_get_bool_option("POCL_MLIR_FORCE_PROGRAM_XCLBIN", false) ||
      pocl_get_bool_option("POCL_MLIR_DISABLE_CMD_BUFFER_FUSION", false)) {
    cl_kernel kernel = kernels[0];
    pocl_cache_program_path(xclbinDir, kernel->program,
                            cmds[0]->program_device_i);
    kernel_xclbin_path = xclbinDir;
    kernel_xclbin_path += "/program.xclbin";
  } else {
    assert(num_kernels == 1 &&
           "AlmaIF MLIR: There can only be one specialized kernel at a time\n");
    pocl_cache_kernel_cachedir_path(xclbinDir, kernels[0]->program,
                                    cmds[0]->program_device_i, kernels[0], "",
                                    cmds[0], specialize);
    kernel_xclbin_path = xclbinDir;
    kernel_xclbin_path += "/parallel.xclbin";
  }

  if (!pocl_exists(kernel_xclbin_path.c_str())) {
    for (int i = 0; i < num_kernels; i++) {
      _cl_command_node *cmd = cmds[i];
      if (cmd->type != CL_COMMAND_NDRANGE_KERNEL) {
        POCL_ABORT("Almaif: trying to compile non-ndrange command\n");
      }
      cl_kernel kernel = kernels[i];
      if (!kernel)
        kernel = cmd->command.run.kernel;
      assert(kernel);

      char cachedir[POCL_MAX_PATHNAME_LENGTH];
      pocl_cache_kernel_cachedir_path(cachedir, kernel->program,
                                      cmd->program_device_i, kernel, "", cmd,
                                      specialize);
      std::string kernel_standard_mlir_path = cachedir;

      // Check whether parallel_hls.cpp already exists
      bool emitOpenCL = pocl_get_bool_option("POCL_MLIR_EMIT_OPENCL", 0);
      std::string kernel_parallel_hlscpp_path;
      if (emitOpenCL) {
        kernel_parallel_hlscpp_path =
            kernel_standard_mlir_path + POCL_PARALLEL_HLSOPENCL_FILENAME;
      } else {
        kernel_parallel_hlscpp_path =
            kernel_standard_mlir_path + POCL_PARALLEL_HLSCPP_FILENAME;
      }

      if (!pocl_exists(kernel_parallel_hlscpp_path.c_str())) {

        kernel_standard_mlir_path += POCL_PARALLEL_MLIR_FILENAME;
        if (!pocl_exists(kernel_standard_mlir_path.c_str())) {
          int error = poclMlirGenerateStandardWorkgroupFunction(
              cmd->program_device_i, device, kernel, cmd, specialize, cachedir);

          POCL_MSG_PRINT_ALMAIF("Generated %s\n",
                                kernel_standard_mlir_path.c_str());
          if (error) {
            POCL_UNLOCK(bd->mlir_compile_lock);
            POCL_ABORT("MLIR: poclMlirGenerateStandardWorkgroupFunction()"
                       " failed for kernel %s\n",
                       kernel->name);
          }
        }

        if (!pocl_offline_compile && d->Dev->isEmulationDevice()) {
          std::string kernel_parallel_llvm_path = cachedir;
          kernel_parallel_llvm_path += POCL_PARALLEL_BC_FILENAME;
          if (!pocl_exists(kernel_parallel_llvm_path.c_str())) {
            int error = poclMlirGenerateLlvmFunction(cmd->program_device_i,
                                                     device, kernel, cmd,
                                                     specialize, cachedir, 1);

            POCL_MSG_PRINT_ALMAIF("Generated %s\n",
                                  kernel_parallel_llvm_path.c_str());
            if (error) {
              POCL_UNLOCK(bd->mlir_compile_lock);
              POCL_ABORT("MLIR: pocl_mlir_generate_llvm_workgroup_function()"
                         " failed for kernel %s\n",
                         kernel->name);
            }
          }

          POCL_UNLOCK(bd->mlir_compile_lock);
          return 0;
        }
        std::string kernel_parallel_mlir_path = cachedir;
        kernel_parallel_mlir_path += "/parallel_hls.mlir";
        if (!pocl_exists(kernel_parallel_mlir_path.c_str())) {
          int error = pocl_mlir_generate_hls_function(
              cmd->program_device_i, device, kernel, cmd, specialize, cachedir);

          if (!pocl_exists(kernel_parallel_mlir_path.c_str())) {
            POCL_ABORT("MLIR: pocl_mlir_generate_hls_function()"
                       " failed for kernel %s\n",
                       kernel->name);
          }
          POCL_MSG_PRINT_ALMAIF("Generated %s\n",
                                kernel_parallel_mlir_path.c_str());
        }
        assert(cmd->command.run.kernel);

        pocl_almaif_mlir_generate_hlscpp_from_hlsmlir(
            kernel_parallel_mlir_path, kernel_parallel_hlscpp_path, emitOpenCL);
        if (!pocl_exists(kernel_parallel_hlscpp_path.c_str())) {
          POCL_ABORT("Failed generating %s\n",
                     kernel_parallel_hlscpp_path.c_str());
        }
      } else {
        POCL_MSG_PRINT_ALMAIF(
            "Vitis c++ file %s exists, skipping regeneration\n",
            kernel_parallel_hlscpp_path.c_str());
      }

      std::string kernel_vitis_zip_path = cachedir;
      kernel_vitis_zip_path +=
          "/kernel_" + std::string(kernel->name) + "_vitis.zip";
      std::string workgroup_function_name = "pocl_mlir_";
      workgroup_function_name += kernel->name;

      std::string kernel_vitis_folder_path = cachedir;
      kernel_vitis_folder_path += "/vitis_project_" + std::string(kernel->name);

      std::string kernel_iprepo_path = cachedir;
      kernel_iprepo_path += "/ip_repo_" + std::string(kernel->name);

      pocl_rm_rf(kernel_vitis_folder_path.c_str());
      pocl_almaif_mlir_generate_vitis_zip_from_hlscpp(cachedir, kernel->name);
      if (!pocl_exists(kernel_vitis_zip_path.c_str())) {
        POCL_ABORT("Failed generating %s\n", kernel_vitis_zip_path.c_str());
      }
      kernelVitisZipPaths.push_back(kernel_vitis_zip_path);
      kernelVitisFolderPaths.push_back(kernel_vitis_folder_path);
      kernelIpRepoPaths.push_back(kernel_iprepo_path);
      kernelNames.push_back(kernel->name);
      workgroupFunctionNames.push_back(workgroup_function_name);
    }

    std::string kernel_xo_path = xclbinDir;
    kernel_xo_path += "/parallel.xo";
    std::string kernel_xo_project_path = xclbinDir;
    kernel_xo_project_path += "/vivado_xo";
    pocl_almaif_mlir_generate_xo_from_rtl(
        kernelVitisZipPaths, kernel_xo_path, kernel_xo_project_path,
        kernelNames, workgroupFunctionNames, kernelVitisFolderPaths,
        kernelIpRepoPaths);
    if (!pocl_exists(kernel_xo_path.c_str())) {
      POCL_ABORT("Failed generating %s\n", kernel_xo_path.c_str());
    }

    pocl_almaif_mlir_generate_xclbin_from_xo(kernel_xo_path, kernel_xclbin_path,
                                             xclbinDir, kernelVitisFolderPaths,
                                             kernelIpRepoPaths);
    if (!pocl_exists(kernel_xclbin_path.c_str())) {
      POCL_ABORT("Failed generating %s\n", kernel_xclbin_path.c_str());
    }
  } else {
    POCL_MSG_PRINT_ALMAIF("XCLBIN %s exists, skipping regeneration\n",
                          kernel_xclbin_path.c_str());
  }

  std::string kernel_firmware_path = xclbinDir;
  kernel_firmware_path += "/firmware.img";
  if (!pocl_exists(kernel_firmware_path.c_str())) {
    POCL_MSG_PRINT_ALMAIF("Starting firmware generation\n");
    std::string kernel_arg_setter_path = xclbinDir;
    kernel_arg_setter_path += "/kernel_arg_setter.h";
    generateLaunchWorkgroupFunction(d, num_kernels, kernels, cmds,
                                    kernel_arg_setter_path, specialize);

    pocl_almaif_mlir_generate_firmware(kernel_firmware_path, xclbinDir);
    if (!pocl_exists(kernel_firmware_path.c_str())) {
      POCL_ABORT("Failed generating %s\n", kernel_firmware_path.c_str());
    }
  } else {
    POCL_MSG_PRINT_ALMAIF("Firmware %s exists, skipping regeneration\n",
                          kernel_firmware_path.c_str());
  }

  if (pocl_get_bool_option("POCL_MLIR_FORCE_PROGRAM_XCLBIN", false) ||
      pocl_get_bool_option("POCL_MLIR_DISABLE_CMD_BUFFER_FUSION", false)) {
    // We have just created the FPGA bitstream with all the kernels.
    // Pre-load one of the kernels in, just to save on reconfiguration time on
    // the first upcoming enqueue. Let's hope the outside doesn't free the
    // kernels[0], since we need to later check that the program matches with
    // the launched kernel!!!  TODO: FIX the leak
    pocl_almaif_compile_kernel(cmds[0], kernels[0], device, 0);
  }

  POCL_UNLOCK(bd->mlir_compile_lock);
  return 0;
}

int pocl_almaif_mlir_compile(_cl_command_node *cmd, cl_kernel kernel,
                             cl_device_id device, int specialize) {
  return pocl_almaif_mlir_compile_kernels(1, &cmd, &kernel, device, specialize);
}

char *pocl_almaif_mlir_init_build(void *data) {
  AlmaifData *D = (AlmaifData *)data;
  mlir_backend_data_t *bd =
      (mlir_backend_data_t *)D->compilationData->backend_data;
  assert(bd);
  return NULL;
}

int pocl_almaif_mlir_device_hash(const char *adf_file, const char *llvm_triplet,
                                 char *output) {
  char tmp[10] = "mlirhash";
  strncpy(output, tmp, 10);
  return 0;
}

int pocl_almaif_mlir_build_source(cl_program program, cl_uint device_i,
                                  cl_uint num_input_headers,
                                  const cl_program *input_headers,
                                  const char **header_include_names,
                                  int link_builtin_lib) {
  assert(program->devices[device_i]->compiler_available == CL_TRUE);
  assert(program->devices[device_i]->linker_available == CL_TRUE);

#ifdef ENABLE_LLVM

  POCL_MSG_PRINT_LLVM("building from sources for device %d\n", device_i);

  return poclMlirBuildProgram(program, device_i, num_input_headers,
                              input_headers, header_include_names,
                              link_builtin_lib);

#else
  POCL_RETURN_ERROR_ON(1, CL_BUILD_PROGRAM_FAILURE,
                       "This device requires LLVM to build from sources\n");
#endif
}

int pocl_almaif_mlir_setup_metadata(cl_device_id device, cl_program program,
                                    unsigned program_device_i) {
  unsigned num_kernels = poclMlirGetKernelCount(program, program_device_i);

  /* TODO zero kernels in program case */
  POCL_MSG_PRINT_ALMAIF("Setting metadata for %d kernels\n", num_kernels);
  if (num_kernels) {
    program->num_kernels = num_kernels;
    program->kernel_meta = (pocl_kernel_metadata_t *)calloc(
        program->num_kernels, sizeof(pocl_kernel_metadata_t));
    poclMlirGetKernelsMetadata(program, program_device_i);
  }
  return 1;
}

cl_int pocl_almaif_mlir_create_finalized_command_buffer(
    cl_device_id device, cl_command_buffer_khr command_buffer) {
  AlmaifData *d = (AlmaifData *)device->data;

  if (!pocl_get_bool_option("POCL_MLIR_DISABLE_CMD_BUFFER_FUSION", false)) {

    _cl_command_node *cmd;
    LL_FOREACH (command_buffer->cmds, cmd) {
      if (cmd->type != CL_COMMAND_NDRANGE_KERNEL) {
        POCL_MSG_WARN(
            "Only NDRange cmds implemented for MLIR command buffers\n");
        return CL_INVALID_OPERATION;
      }

      cmd->command.run.force_large_grid_wg_func = 1;

      int specialize = 1;
      cl_kernel kernel = cmd->command.run.kernel;
      char cachedir[POCL_MAX_PATHNAME_LENGTH];
      pocl_cache_kernel_cachedir_path(
          cachedir, kernel->program, cmd->program_device_i, kernel, "", cmd, 1);
      int error = poclMlirGenerateStandardWorkgroupFunction(
          cmd->program_device_i, device, cmd->command.run.kernel, cmd,
          specialize, cachedir);

      if (error) {
        POCL_ABORT("MLIR: poclMlirGenerateStandardWorkgroupFunction()"
                   " failed for cmd buffer subkernel %s\n",
                   cmd->command.run.kernel->name);
      }
    }
    const char empty_suffix[1] = "";
    pocl_mlir_generate_cmd_buffer_function(device, command_buffer,
                                           empty_suffix);

    POCL_MSG_PRINT_ALMAIF("Generated cmd buffer function, compiling it now\n");

    cl_kernel kernel = command_buffer->megaKernel;
    _cl_command_node fake_cmd;
    memset(&fake_cmd, 0, sizeof(_cl_command_node));
    fake_cmd.program_device_i = device->dev_id;
    fake_cmd.device = device;
    fake_cmd.type = CL_COMMAND_NDRANGE_KERNEL;
    fake_cmd.command.run.kernel = kernel;
    fake_cmd.command.run.hash = calloc(1, sizeof(pocl_kernel_hash_t));
    fake_cmd.command.run.pc.local_size[0] = 1;
    fake_cmd.command.run.pc.local_size[1] = 1;
    fake_cmd.command.run.pc.local_size[2] = 1;
    fake_cmd.command.run.pc.work_dim = 1;
    fake_cmd.command.run.pc.num_groups[0] = 1;
    fake_cmd.command.run.pc.num_groups[1] = 1;
    fake_cmd.command.run.pc.num_groups[2] = 1;
    fake_cmd.command.run.pc.global_offset[0] = 0;
    fake_cmd.command.run.pc.global_offset[1] = 0;
    fake_cmd.command.run.pc.global_offset[2] = 0;

    pocl_almaif_compile_kernel(&fake_cmd, kernel, device, 1);

    POCL_MSG_PRINT_ALMAIF("Cmd buffer function compiled\n");

    free(fake_cmd.command.run.hash);
    return CL_SUCCESS;
  } else {
    int num_kernels = 0;
    _cl_command_node *cmd = nullptr;
    LL_FOREACH (command_buffer->cmds, cmd) {
      if (cmd->type != CL_COMMAND_NDRANGE_KERNEL) {
        POCL_MSG_WARN(
            "Only NDRange cmds implemented for MLIR command buffers\n");
        return CL_INVALID_OPERATION;
      }

      cmd->command.run.force_large_grid_wg_func = 1;

      int specialize = 1;
      cl_kernel kernel = cmd->command.run.kernel;
      char cachedir[POCL_MAX_PATHNAME_LENGTH];
      pocl_cache_kernel_cachedir_path(
          cachedir, kernel->program, cmd->program_device_i, kernel, "", cmd, 1);
      int error = poclMlirGenerateStandardWorkgroupFunction(
          cmd->program_device_i, device, cmd->command.run.kernel, cmd,
          specialize, cachedir);

      if (error) {
        POCL_ABORT("MLIR: poclMlirGenerateStandardWorkgroupFunction()"
                   " failed for cmd buffer subkernel %s\n",
                   cmd->command.run.kernel->name);
      }
      num_kernels++;
    }
    command_buffer->fake_single_kernel_cmd_buffers =
        (cl_command_buffer_khr *)calloc(num_kernels,
                                        sizeof(cl_command_buffer_khr));
    cl_kernel *tmp_kernels =
        (cl_kernel *)calloc(num_kernels, sizeof(cl_kernel));
    _cl_command_node *tmp_cmds =
        (_cl_command_node *)calloc(num_kernels, sizeof(_cl_command_node));
    _cl_command_node **tmp_cmd_ptrs =
        (_cl_command_node **)calloc(num_kernels, sizeof(_cl_command_node *));

    int idx = 0;
    cmd = nullptr;
    _cl_command_node *_tmp;
    LL_FOREACH_SAFE (command_buffer->cmds, cmd, _tmp) {
      cl_int err;
      command_buffer->fake_single_kernel_cmd_buffers[idx] =
          clCreateCommandBufferKHR(command_buffer->num_queues,
                                   command_buffer->queues,
                                   command_buffer->properties, &err);
      if (err != CL_SUCCESS) {
        POCL_ABORT("failed creating fake command buffer\n");
      }
      cl_command_buffer_khr tmp_cmd_buf =
          command_buffer->fake_single_kernel_cmd_buffers[idx];
      size_t tmp_global_size[3] = {0, 0, 0};
      int tmp_work_dim = cmd->command.run.pc.work_dim;
      tmp_global_size[0] =
          cmd->command.run.pc.num_groups[0] * cmd->command.run.pc.local_size[0];
      tmp_global_size[1] = (tmp_work_dim > 1)
                               ? (cmd->command.run.pc.num_groups[1] *
                                  cmd->command.run.pc.local_size[1])
                               : 1;
      tmp_global_size[2] = (tmp_work_dim > 2)
                               ? (cmd->command.run.pc.num_groups[2] *
                                  cmd->command.run.pc.local_size[2])
                               : 1;
      cl_command_properties_khr *command_props = NULL;
      cl_mutable_command_khr *mutable_cmd = NULL;
      if (command_buffer->is_mutable) {
        // Reconstruct the command properties
        command_props = (cl_command_properties_khr *)calloc(
            3, sizeof(cl_command_properties_khr));
        command_props[0] = CL_MUTABLE_DISPATCH_UPDATABLE_FIELDS_KHR;
        bool mutable_args = cmd->command.run.updatable_fields &
                            CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;
        bool mutable_global_size = cmd->command.run.updatable_fields &
                                   CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR;
        command_props[1] = 0;
        if (mutable_args)
          command_props[1] |= CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;
        if (mutable_global_size)
          command_props[1] |= CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR;
        command_props[2] = 0;
      }
      err = clCommandNDRangeKernelKHR(
          tmp_cmd_buf, command_buffer->queues[cmd->queue_idx], command_props,
          cmd->command.run.kernel, cmd->command.run.pc.work_dim,
          cmd->command.run.pc.global_offset, tmp_global_size,
          cmd->command.run.pc.local_size, 0, NULL, NULL, mutable_cmd);
      if (err != CL_SUCCESS) {
        POCL_ABORT("failed creating fake command ndrange kernel\n");
      }
      char suffix[64];
      snprintf(suffix, 64, "_node%i", idx);
      tmp_cmd_buf->cmds[0].device = device;
      tmp_cmd_buf->cmds[0].command.run.force_large_grid_wg_func = 1;
      pocl_mlir_generate_cmd_buffer_function(device, tmp_cmd_buf, suffix);

      POCL_MSG_PRINT_ALMAIF(
          "Generated cmd buffer function, compiling it now\n");

      tmp_cmds[idx].program_device_i = device->dev_id;
      tmp_cmds[idx].device = device;
      tmp_cmds[idx].type = CL_COMMAND_NDRANGE_KERNEL;
      tmp_cmds[idx].command.run.kernel = tmp_cmd_buf->megaKernel;
      tmp_cmds[idx].command.run.hash = calloc(1, sizeof(pocl_kernel_hash_t));
      tmp_cmds[idx].command.run.pc.local_size[0] = 1;
      tmp_cmds[idx].command.run.pc.local_size[1] = 1;
      tmp_cmds[idx].command.run.pc.local_size[2] = 1;
      tmp_cmds[idx].command.run.pc.work_dim = 1;
      tmp_cmds[idx].command.run.pc.num_groups[0] = 1;
      tmp_cmds[idx].command.run.pc.num_groups[1] = 1;
      tmp_cmds[idx].command.run.pc.num_groups[2] = 1;
      tmp_cmds[idx].command.run.pc.global_offset[0] = 0;
      tmp_cmds[idx].command.run.pc.global_offset[1] = 0;
      tmp_cmds[idx].command.run.pc.global_offset[2] = 0;
      tmp_kernels[idx] = tmp_cmd_buf->megaKernel;
      tmp_cmd_ptrs[idx] = &tmp_cmds[idx];
      tmp_cmd_buf->megaKernel->higher_level_cmd_buf = command_buffer;
      idx++;
    }

    pocl_almaif_mlir_compile_kernels(num_kernels, tmp_cmd_ptrs, tmp_kernels,
                                     device, 1);

    POCL_MSG_PRINT_ALMAIF(
        "Specialized kernels, not in a cmd buffer compiled\n");
    free(tmp_kernels);
    free(tmp_cmd_ptrs);
    free(tmp_cmds);

    // free(fake_cmd.command.run.hash);
    return CL_SUCCESS;
  }
}

cl_int
pocl_almaif_mlir_free_command_buffer(cl_device_id device,
                                     cl_command_buffer_khr command_buffer) {

  return pocl_mlir_free_command_buffer(device, command_buffer);
}

cl_int pocl_almaif_mlir_run_command_buffer(cl_device_id device,
                                           cl_command_buffer_khr command_buffer,
                                           cl_uint num_events_in_wait_list,
                                           const cl_event *event_wait_list,
                                           cl_event *event) {

  const size_t globalSize[1] = {1};
  const size_t localSize[1] = {1};

  if (pocl_get_bool_option("POCL_MLIR_DISABLE_CMD_BUFFER_FUSION", false)) {
    int idx = 0;
    _cl_command_node *cmd;
    LL_FOREACH (command_buffer->cmds, cmd) {
      cl_int status =
          (clEnqueueNDRangeKernel)(command_buffer
                                       ->fake_single_kernel_cmd_buffers[idx]
                                       ->queues[0],
                                   command_buffer
                                       ->fake_single_kernel_cmd_buffers[idx]
                                       ->megaKernel,
                                   1, NULL, globalSize, localSize,
                                   num_events_in_wait_list, event_wait_list,
                                   event);
      if (status != CL_SUCCESS) {
        return status;
      }
      idx++;
    }
    return CL_SUCCESS;
  } else {
    cl_int status =
        (clEnqueueNDRangeKernel)(command_buffer->queues[0],
                                 command_buffer->megaKernel, 1, NULL,
                                 globalSize, localSize, num_events_in_wait_list,
                                 event_wait_list, event);
    return status;
  }
}
