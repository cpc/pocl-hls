/* pocl_mlir_cmd_buffer.cc: Generate mega-function out of the command buffer,

   Copyright (c) 2025 Topi Leppänen / Tampere University

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include "pocl.h"
#include "pocl_cache.h"
#include "pocl_cl.h"
#include "pocl_file_util.h"
#include "pocl_llvm_api.h"
#include "pocl_mlir.h"
#include "pocl_mlir_file_util.hh"
#include "pocl_mlir_passes.hh"
#include "utlist.h"

#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/LLVMCommon/LoweringOptions.h>
#include <mlir/Conversion/VectorToSCF/VectorToSCF.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/Linalg/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <vector>

mlir::Type replaceDynamicMemrefArgWithConstantRange(mlir::func::FuncOp func,
                                                    int arg_index,
                                                    size_t size_in_bytes) {
  mlir::Type argType = func.getArgument(arg_index).getType();

  if (auto memrefType = mlir::dyn_cast_or_null<mlir::MemRefType>(argType)) {
    size_t elementSize =
        memrefType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t numElements =
        (size_in_bytes + elementSize - 1) / elementSize; // rounding up

    // Create a new MemRefType with a static size
    auto staticSizedMemrefType = mlir::MemRefType::get(
        {numElements}, memrefType.getElementType(), memrefType.getLayout(),
        memrefType.getMemorySpace());

    func.getArgument(arg_index).setType(staticSizedMemrefType);

    // Update the function type
    auto funcType = func.getFunctionType();
    llvm::SmallVector<mlir::Type, 4> newArgTypes(funcType.getInputs().begin(),
                                                 funcType.getInputs().end());
    newArgTypes[arg_index] = staticSizedMemrefType;
    auto newFuncType = mlir::FunctionType::get(func.getContext(), newArgTypes,
                                               funcType.getResults());
    func.setType(newFuncType);
    return staticSizedMemrefType;
  }
  return argType;
}

struct cmd_buffer_arg_aliasing_t {
  cl_mem mem_ptr;
  int kernel_idx;
  int local_arg_idx;
  int linear_arg_idx;
  bool is_scalar;
};
typedef std::vector<std::vector<cmd_buffer_arg_aliasing_t>> AliasingGroups_t;

int pocl_mlir_generate_cmd_buffer_function_nowrite(
    cl_device_id device, cl_command_buffer_khr command_buffer,
    mlir::OwningOpRef<mlir::ModuleOp> &CommandBufferModule,
    mlir::MLIRContext *MLIRContext) {
  pocl_kernel_metadata_t *megaMeta = command_buffer->megaKernel->meta;
  assert(megaMeta);
  _cl_command_node *cmd;

  int num_of_kernels = 0;
  LL_FOREACH (command_buffer->cmds, cmd) {
    if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
      num_of_kernels++;
  }
  command_buffer->num_of_settable_arguments =
      (int *)calloc(num_of_kernels, sizeof(int));

  AliasingGroups_t *AliasingGroups =
      new std::vector<std::vector<cmd_buffer_arg_aliasing_t>>();
  mlir::SmallVector<mlir::Type, 64> combinedArgTypes;
  mlir::SmallVector<mlir::func::FuncOp, 16> clonedFunctions;
  mlir::SmallVector<int, 64> argIndices;
  int total_settable_arg_counter = 0;
  int cmd_idx = 0;
  printf("### HELLO WE GOT HERE calling generate_cmd_buffer_function\n");
  LL_FOREACH (command_buffer->cmds, cmd) {
    int settable_arg_counter = 0;
    if (cmd->type != CL_COMMAND_NDRANGE_KERNEL) {
      POCL_MSG_WARN(
          "Non ndrange command when trying to compile command buffer\n");
      return false;
    }
    cl_kernel Kernel = cmd->command.run.kernel;
    if (!Kernel) {
      POCL_MSG_WARN("No kernel found for command buffers ndrange cmd\n");
      return false;
    }

    bool mutable_args = false;
    bool mutable_global_size = false;
    if (command_buffer->is_mutable) {
      mutable_args =
          cmd->command.run.updatable_fields & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;
      mutable_global_size = cmd->command.run.updatable_fields &
                            CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR;
    }
    char subModulePath[POCL_MAX_PATHNAME_LENGTH];
    pocl_cache_kernel_cachedir_path(subModulePath, Kernel->program,
                                    cmd->program_device_i, Kernel,
                                    POCL_PARALLEL_MLIR_FILENAME, cmd, 1);
    if (!pocl_exists(subModulePath)) {
      POCL_MSG_WARN(
          "No compiled implementation found for cmd buffer's kernel: %s\n",
          subModulePath);
      return false;
    }

    mlir::OwningOpRef<mlir::ModuleOp> mod;
    if (pocl::mlir::openFile(subModulePath, MLIRContext, mod)) {
      POCL_ABORT("Can't parse program.mlir file in generate workgroup func\n");
    }
    auto wgName = "pocl_mlir_" + std::string(Kernel->name);
    std::vector<mlir::func::FuncOp> funcsToDelete;
    auto funcSym = mod->lookupSymbol<mlir::func::FuncOp>(wgName);
    pocl_kernel_metadata_t *meta = Kernel->meta;
    for (int i = 0; i < meta->num_args; ++i) {
      auto al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL(meta->arg_info[i])) {
        auto _argType =
            replaceDynamicMemrefArgWithConstantRange(funcSym, i, al->size);
      } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
        assert(al->is_raw_ptr == 0); // only raw pointers supported
        cl_mem cl_mem_ptr = *(cl_mem *)(al->value);
        auto argType = replaceDynamicMemrefArgWithConstantRange(
            funcSym, i, cl_mem_ptr->size);
        auto it = AliasingGroups->begin();
        for (; it != AliasingGroups->end(); it++) {
          if (!(it->front().is_scalar) && cl_mem_ptr == it->front().mem_ptr) {
            break;
          }
        }
        if (it != AliasingGroups->end()
            //&& (!mutable_args)) // TODO: Commenting this out makes the buffer
            // arguments not completely mutable.
            // This is currently needed, because of buffer aliasing rules make
            // mutating the buffers quite complicated. Basically we cannot just
            // duplicate buffer args, because at the output of the tool we
            // cannot have aliasing buffers. (OpenCL C output, or HLS C++ output
            // can only take non-aliasing buffers) So the current solutiong
            // deduplicates aliasing (same) buffer args. This is also wrong,
            // since the user should be able to mutate them individually. (So go
            // from same buffer arg given to 2 commands, to then giving
            // different buffer args to the 2 commands in their
            // updateMutable-call). The current limitation is that the user must
            // update all the buffer args of a single AliasingGroup in the same
            // updateMutables-call.
        ) {
          // Reuse the same buffer for different kernels in a command buffer
          argIndices.push_back(it->front().linear_arg_idx);
          // mem_ptr and linear_arg idx are always the same for each aliasing
          // buffer
          it->push_back({it->front().mem_ptr, cmd_idx, i,
                         it->front().linear_arg_idx, false});
        } else {
          // Two cases:
          // 1. The buffer has not been used before (in non-mutable CB)
          // 2. In case of mutable buffers, no reuse is allowed in case user
          // eventually switches the buffer arg.
          combinedArgTypes.push_back(argType);
          argIndices.push_back(total_settable_arg_counter);
          AliasingGroups->push_back(
              {{cl_mem_ptr, cmd_idx, i, total_settable_arg_counter, false}});
          settable_arg_counter++;
          total_settable_arg_counter++;
          // Initialize a new AliasingGroup with the current argument as the
          // "real" settable argument
        }
      } else {
        // The argument is a scalar. In case of non-mutable buffer, we will
        // later bake these in as constants, so no separate argument needed for
        // them
        if (mutable_args) {
          // For mutable cmds, we want to preserve scalar args independently
          combinedArgTypes.push_back(funcSym.getArgumentTypes()[i]);
          argIndices.push_back(total_settable_arg_counter);
          AliasingGroups->push_back(
              {{NULL, cmd_idx, i, total_settable_arg_counter, true}});
          settable_arg_counter++;
          total_settable_arg_counter++;
        }
      }
    }
    if (mutable_global_size) {
      // Push the num_groups for every kernel as a hidden argument
      for (auto dim : {0, 1, 2}) {
        combinedArgTypes.push_back(mlir::IndexType::get(MLIRContext));
        argIndices.push_back(total_settable_arg_counter);
        settable_arg_counter++;
        total_settable_arg_counter++;
      }
    }

    if (!CommandBufferModule) {
      CommandBufferModule =
          mlir::ModuleOp::create(mlir::UnknownLoc::get(MLIRContext));
    }
    auto alreadyExistingSym =
        CommandBufferModule->lookupSymbol<mlir::func::FuncOp>(wgName);
    if (!alreadyExistingSym) {
      mlir::func::FuncOp func = funcSym.clone();
      CommandBufferModule->push_back(func);
      clonedFunctions.push_back(func);
    } else {
      clonedFunctions.push_back(alreadyExistingSym);
    }
    command_buffer->num_of_settable_arguments[cmd_idx] = settable_arg_counter;
    cmd_idx++;
  }

  megaMeta->num_args = total_settable_arg_counter;
  megaMeta->arg_info = (pocl_argument_info *)malloc(sizeof(pocl_argument_info) *
                                                    megaMeta->num_args);
  command_buffer->megaKernel->dyn_arguments = (pocl_argument *)calloc(
      total_settable_arg_counter, sizeof(struct pocl_argument));

  mlir::PassManager PM(MLIRContext);
  mlir::OpPassManager &optPM = PM.nest<mlir::func::FuncOp>();
  PM.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(PM.run(*CommandBufferModule))) {
    POCL_MSG_PRINT_LLVM("Failed running the MLIR compiler passes 1\n");
  }

  mlir::FunctionType newFuncType =
      mlir::FunctionType::get(MLIRContext, combinedArgTypes, {});
  int newFuncNameLen = 11 + strlen(command_buffer->megaKernel->name);
  char *newFuncName = (char *)calloc(1, newFuncNameLen);
  snprintf(newFuncName, newFuncNameLen, "pocl_mlir_%s",
           command_buffer->megaKernel->name);
  mlir::func::FuncOp newFunc = mlir::func::FuncOp::create(
      mlir::UnknownLoc::get(MLIRContext), newFuncName, newFuncType);
  auto unitAttr = mlir::UnitAttr::get(MLIRContext);
  newFunc->setAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName(), unitAttr);
  mlir::IntegerType IntType = mlir::IntegerType::get(MLIRContext, 64);
  newFunc->setAttr("CL_arg_count",
                   mlir::IntegerAttr::get(IntType, total_settable_arg_counter));
  newFunc->setAttr("CL_command_buffer", unitAttr);
  CommandBufferModule->push_back(newFunc);

  mlir::Block *entryBlock = newFunc.addEntryBlock();
  mlir::OpBuilder builder(MLIRContext);
  builder.setInsertionPointToStart(entryBlock);

  unsigned argIndex = 0;
  int idx = 0;
  LL_FOREACH (command_buffer->cmds, cmd) {

    bool mutable_args = false;
    bool mutable_global_size = false;
    if (command_buffer->is_mutable) {
      mutable_args =
          cmd->command.run.updatable_fields & CL_MUTABLE_DISPATCH_ARGUMENTS_KHR;
      mutable_global_size = cmd->command.run.updatable_fields &
                            CL_MUTABLE_DISPATCH_GLOBAL_SIZE_KHR;
    }

    auto clonedFunction = clonedFunctions[idx];
    mlir::SmallVector<mlir::Value, 64> args;
    for (auto i = 0; i < clonedFunction.getNumArguments() - (3 + 15); i++) {
      auto argType = clonedFunction.getArgumentTypes()[i];
      pocl_kernel_metadata_t *meta = cmd->command.run.kernel->meta;
      auto al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL(meta->arg_info[i])) {
        auto argTypeMemref = mlir::dyn_cast_or_null<mlir::MemRefType>(argType);
        assert(argTypeMemref);
        auto localArg = mlir::memref::AllocaOp::create(
            builder, builder.getUnknownLoc(), argTypeMemref);
        args.push_back(localArg);
      } else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER) {
        int megaKernelArgIndex = argIndices[argIndex];
        args.push_back(entryBlock->getArgument(megaKernelArgIndex));
        megaMeta->arg_info[megaKernelArgIndex] = meta->arg_info[i];
        command_buffer->megaKernel->dyn_arguments[megaKernelArgIndex] = *al;
        argIndex++;
      } else {
        if (mutable_args) {
          int megaKernelArgIndex = argIndices[argIndex];
          args.push_back(entryBlock->getArgument(megaKernelArgIndex));
          megaMeta->arg_info[megaKernelArgIndex] = meta->arg_info[i];
          command_buffer->megaKernel->dyn_arguments[megaKernelArgIndex] = *al;
          // Deep-copy the scalar value so it survives beyond the original
          // kernel's lifetime.
          if (al->value && al->size > 0) {
            void *copy = malloc(al->size);
            memcpy(copy, al->value, al->size);
            command_buffer->megaKernel->dyn_arguments[megaKernelArgIndex]
                .value = copy;
          }
          argIndex++;
        } else {
          // In case of non-mutable CB, we will specialize by the scalar arg
          // value
          uint64_t const_val = 0;
          memcpy(&const_val, al->value, al->size);
          mlir::Value constValue;
          if (mlir::isa<mlir::IntegerType>(argType)) {
            constValue = mlir::arith::ConstantIntOp::create(
                builder, builder.getUnknownLoc(), argType, const_val);
          } else if (auto argType_float =
                         mlir::dyn_cast<mlir::FloatType>(argType)) {
            switch (al->size) {
            case 4: {
              float value = *reinterpret_cast<float *>(al->value);
              mlir::APFloat apFloat = mlir::APFloat(value);
              constValue = mlir::arith::ConstantFloatOp::create(
                  builder, builder.getUnknownLoc(), argType_float, apFloat);
            } break;
            case 8: {
              double value = *reinterpret_cast<double *>(al->value);
              mlir::APFloat apFloat = mlir::APFloat(value);
              constValue = mlir::arith::ConstantFloatOp::create(
                  builder, builder.getUnknownLoc(), argType_float, apFloat);
            } break;
            default:
              POCL_MSG_WARN("Unknown float size, ignoring\n");
            }
          } else {
            POCL_MSG_WARN(
                "Unknown arg type in cmd buffer compilation, ignoring\n");
          }
          args.push_back(constValue);
        }
      }
    }
    // Create the 3D loop executing all WGs
    mlir::AffineMap lbMapX = mlir::AffineMap::getConstantMap(0, MLIRContext);
    mlir::AffineMap lbMapY = mlir::AffineMap::getConstantMap(0, MLIRContext);
    mlir::AffineMap lbMapZ = mlir::AffineMap::getConstantMap(0, MLIRContext);
    llvm::SmallVector<mlir::AffineMap> ubMaps = {};
    llvm::SmallVector<mlir::Value> upperBoundValues = {};
    for (auto dim : {0, 1, 2}) {
      if (mutable_global_size) {
        int megaKernelArgIndex = argIndices[argIndex + dim];
        pocl_argument_info *numGroupsHiddenArgInfo =
            &megaMeta->arg_info[megaKernelArgIndex];
        numGroupsHiddenArgInfo->name = strdup("hidden_num_groups");
        numGroupsHiddenArgInfo->type = POCL_ARG_TYPE_NONE;
        numGroupsHiddenArgInfo->type_size = sizeof(size_t);
        numGroupsHiddenArgInfo->type_name = strdup("size_t");
        pocl_argument *numGroupsHiddenArgument =
            &command_buffer->megaKernel->dyn_arguments[megaKernelArgIndex];
        numGroupsHiddenArgument->is_set = 1;
        numGroupsHiddenArgument->size = sizeof(size_t);
        numGroupsHiddenArgument->value =
            calloc(1, numGroupsHiddenArgument->size);
        memcpy(numGroupsHiddenArgument->value,
               &cmd->command.run.pc.num_groups[dim],
               numGroupsHiddenArgument->size);

        mlir::AffineExpr d0 = builder.getAffineDimExpr(dim);
        auto ubMap = mlir::AffineMap::get(3, 0, {d0}, MLIRContext);
        ubMaps.push_back(ubMap);
        auto totalNumOfArguments = args.size();
        mlir::Value numGroups = entryBlock->getArgument(megaKernelArgIndex);
        upperBoundValues.push_back(numGroups);
      } else {
        ubMaps.push_back(mlir::AffineMap::getConstantMap(
            cmd->command.run.pc.num_groups[dim], MLIRContext));
      }
    }

    int64_t step = 1;
    mlir::affine::AffineParallelOp affineParallelOp =
        mlir::affine::AffineParallelOp::create(
            builder, builder.getUnknownLoc(), mlir::TypeRange(),
            llvm::ArrayRef<mlir::arith::AtomicRMWKind>(),
            llvm::ArrayRef<mlir::AffineMap>{lbMapX, lbMapY, lbMapZ},
            mlir::ValueRange(), ubMaps, mlir::ValueRange(upperBoundValues),
            llvm::ArrayRef<int64_t>{step, step, step});
    mlir::Block *body = affineParallelOp.getBody();
    mlir::OpBuilder innerBuilder(body, body->begin());
    for (auto dim : {0, 1, 2}) {
      if (mutable_global_size) {
        mlir::Value numGroups = entryBlock->getArgument(argIndices[argIndex]);
        argIndex++;
        args.push_back(numGroups);
      } else {
        auto num_groups = cmd->command.run.pc.num_groups[dim];
        auto num_groups_idx = mlir::arith::ConstantIndexOp::create(
            innerBuilder, innerBuilder.getUnknownLoc(), num_groups);
        args.push_back(num_groups_idx);
      }
    }
    for (auto dim : {0, 1, 2}) {
      auto global_offset = cmd->command.run.pc.global_offset[dim];
      auto global_offset_idx = mlir::arith::ConstantIndexOp::create(
          innerBuilder, innerBuilder.getUnknownLoc(), global_offset);
      args.push_back(global_offset_idx);
    }
    for (auto dim : {0, 1, 2}) {
      auto local_size = cmd->command.run.pc.local_size[dim];
      auto local_size_idx = mlir::arith::ConstantIndexOp::create(
          innerBuilder, innerBuilder.getUnknownLoc(), local_size);
      args.push_back(local_size_idx);
    }
    auto printf_buffer_idx = mlir::arith::ConstantIndexOp::create(
        innerBuilder, innerBuilder.getUnknownLoc(),
        (int64_t)cmd->command.run.pc.printf_buffer);
    args.push_back(printf_buffer_idx);
    auto printf_buffer_position_idx = mlir::arith::ConstantIndexOp::create(
        innerBuilder, innerBuilder.getUnknownLoc(),
        (int64_t)cmd->command.run.pc.printf_buffer_position);
    args.push_back(printf_buffer_position_idx);
    auto printf_buffer_capacity_i32 = mlir::arith::ConstantIntOp::create(
        innerBuilder, innerBuilder.getUnknownLoc(),
        innerBuilder.getIntegerType(32),
        cmd->command.run.pc.printf_buffer_capacity);
    args.push_back(printf_buffer_capacity_i32);
    auto global_var_buffer_idx = mlir::arith::ConstantIndexOp::create(
        innerBuilder, innerBuilder.getUnknownLoc(),
        (int64_t)cmd->command.run.pc.global_var_buffer);
    args.push_back(global_var_buffer_idx);
    auto work_dim_i32 = mlir::arith::ConstantIntOp::create(
        innerBuilder, innerBuilder.getUnknownLoc(),
        innerBuilder.getIntegerType(32), cmd->command.run.pc.work_dim);
    args.push_back(work_dim_i32);
    auto execution_failed_i32 = mlir::arith::ConstantIntOp::create(
        innerBuilder, innerBuilder.getUnknownLoc(),
        innerBuilder.getIntegerType(32), cmd->command.run.pc.execution_failed);
    args.push_back(execution_failed_i32);

    for (auto dim : {0, 1, 2}) {
      auto group_id = affineParallelOp.getRegion().front().getArgument(dim);
      args.push_back(group_id);
    }

    mlir::func::CallOp::create(innerBuilder, builder.getUnknownLoc(),
                               clonedFunction, args);
    clonedFunction->removeAttr(mlir::gpu::GPUDialect::getKernelFuncAttrName());

    idx++;
  }
  mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
  // std::cerr << "AFTER MEGAFUNC" << std::endl;
  // CommandBufferModule->dump();
  // std::cerr << "DUMP DONE" << std::endl;
  command_buffer->aliasing_data = (void *)AliasingGroups;

  return true;
}

int pocl_mlir_generate_cmd_buffer_function(cl_device_id device,
                                           cl_command_buffer_khr command_buffer,
                                           const char *kernel_name_suffix) {

  command_buffer->megaProgram =
      (cl_program)calloc(1, sizeof(struct _cl_program));
  POCL_INIT_OBJECT(command_buffer->megaProgram,
                   command_buffer->queues[0]->context);
  command_buffer->megaProgram->context = command_buffer->queues[0]->context;
  command_buffer->megaProgram->num_devices = 1;
  command_buffer->megaProgram->devices =
      (cl_device_id *)calloc(1, sizeof(cl_device_id));
  command_buffer->megaProgram->devices[0] = device;

  command_buffer->megaKernel = (cl_kernel)calloc(1, sizeof(struct _cl_kernel));
  POCL_INIT_OBJECT(command_buffer->megaKernel, command_buffer->megaProgram);
  size_t suffix_len = strnlen(kernel_name_suffix, POCL_MAX_FILENAME_LENGTH);
  command_buffer->megaKernel->name = (const char *)calloc(1, 20 + suffix_len);
  snprintf((char *)command_buffer->megaKernel->name, 20 + suffix_len,
           "command_buffer%s", kernel_name_suffix);
  command_buffer->megaKernel->context = command_buffer->queues[0]->context;
  command_buffer->megaKernel->program = command_buffer->megaProgram;
  command_buffer->megaKernel->data = (void **)calloc(1, sizeof(void *));
  device->ops->create_kernel(device, command_buffer->megaProgram,
                             command_buffer->megaKernel, 0);

  PoclLLVMContextData *PoCLLLVMContext =
      (PoclLLVMContextData *)
          command_buffer->megaKernel->context->llvm_context_data;
  mlir::MLIRContext *MLIRContext = PoCLLLVMContext->MLIRContext;

  // Fake run command just to get the final cache specialized cache dir,
  // we are always going create a specialized kernel instance from the cmd
  // buffer
  _cl_command_node fake_cmd_buffer_cmd;
  memset(&fake_cmd_buffer_cmd, 0, sizeof(_cl_command_node));
  fake_cmd_buffer_cmd.program_device_i = device->dev_id;
  fake_cmd_buffer_cmd.device = device;
  fake_cmd_buffer_cmd.type = CL_COMMAND_NDRANGE_KERNEL;
  fake_cmd_buffer_cmd.command.run.kernel = command_buffer->megaKernel;
  fake_cmd_buffer_cmd.command.run.hash = calloc(1, sizeof(pocl_kernel_hash_t));
  fake_cmd_buffer_cmd.command.run.pc.local_size[0] = 1;
  fake_cmd_buffer_cmd.command.run.pc.local_size[1] = 1;
  fake_cmd_buffer_cmd.command.run.pc.local_size[2] = 1;
  fake_cmd_buffer_cmd.command.run.pc.work_dim = 1;
  fake_cmd_buffer_cmd.command.run.pc.execution_failed = 0;
  fake_cmd_buffer_cmd.command.run.pc.num_groups[0] = 1;
  fake_cmd_buffer_cmd.command.run.pc.num_groups[1] = 1;
  fake_cmd_buffer_cmd.command.run.pc.num_groups[2] = 1;
  fake_cmd_buffer_cmd.command.run.pc.global_offset[0] = 0;
  fake_cmd_buffer_cmd.command.run.pc.global_offset[1] = 0;
  fake_cmd_buffer_cmd.command.run.pc.global_offset[2] = 0;

  pocl_cache_set_cmd_buffer_hash(command_buffer);

  command_buffer->megaKernel->meta =
      (pocl_kernel_metadata_t *)calloc(1, sizeof(pocl_kernel_metadata_t));
  command_buffer->megaProgram->kernel_meta = command_buffer->megaKernel->meta;
  command_buffer->megaKernel->meta->build_hash =
      (pocl_kernel_hash_t *)calloc(1, sizeof(pocl_kernel_hash_t));
  memcpy(command_buffer->megaKernel->meta->build_hash,
         command_buffer->megaProgram->build_hash, sizeof(pocl_kernel_hash_t));

  // memcpy(fake_cmd_buffer_cmd.command.run.hash,
  // command_buffer->megaProgram->build_hash, sizeof(pocl_kernel_hash_t));
  char cachedir[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_kernel_cachedir_path(cachedir, command_buffer->megaProgram, 0,
                                  command_buffer->megaKernel, "",
                                  &fake_cmd_buffer_cmd, 1);

  char CmdBufferFuncPath[POCL_MAX_PATHNAME_LENGTH];
  pocl_cache_kernel_cachedir_path(
      CmdBufferFuncPath, command_buffer->megaProgram, 0,
      command_buffer->megaKernel, POCL_PARALLEL_MLIR_FILENAME,
      &fake_cmd_buffer_cmd, 1);

  mlir::OwningOpRef<mlir::ModuleOp> Module;
  bool succeeded = pocl_mlir_generate_cmd_buffer_function_nowrite(
      device, command_buffer, Module, MLIRContext);
  if (!succeeded) {
    POCL_MSG_ERR("Failed generating mlir command buffer\n");
    return CL_FAILED;
  }

  if (pocl::mlir::runAffinePasses(Module, false, true) == CL_FAILED) {
    POCL_MSG_PRINT_LLVM("Failed running the MLIR affine cmd buffer pass\n");
    return CL_FAILED;
  }

  // Remove all other unused functions, except the pocl_mlir_command_buffer
  mlir::PassManager pm(MLIRContext);
  pm.addPass(mlir::createSymbolPrivatizePass(
      llvm::ArrayRef<std::string>{"pocl_mlir_command_buffer"}));
  pm.addPass(mlir::createSymbolDCEPass());
  if (mlir::failed(pm.run(*Module))) {
    POCL_MSG_PRINT_LLVM("Failed removing unused funcs\n");
    return CL_FAILED;
  }

  // std::cerr << "Dumping Affine cmd buffer pass output" << std::endl;
  // Module->dump();
  // std::cerr << "Dump done" << std::endl;

  char dir_path[POCL_MAX_PATHNAME_LENGTH];
  int Error = pocl_mkdir_p(cachedir);
  if (Error) {
    POCL_MSG_PRINT_GENERAL("Unable to create directory %s.\n", dir_path);
    return Error;
  }

  free(fake_cmd_buffer_cmd.command.run.hash);
  return pocl::mlir::writeOutput(Module, CmdBufferFuncPath);
}

cl_int pocl_mlir_free_command_buffer(cl_device_id device,
                                     cl_command_buffer_khr command_buffer) {

  if (pocl_get_bool_option("POCL_MLIR_DISABLE_CMD_BUFFER_FUSION", false)) {
    int idx = 0;
    _cl_command_node *cmd;
    LL_FOREACH (command_buffer->cmds, cmd) {
      device->ops->free_kernel(
          device,
          command_buffer->fake_single_kernel_cmd_buffers[idx]->megaProgram,
          command_buffer->fake_single_kernel_cmd_buffers[idx]->megaKernel, 0);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]
               ->megaKernel->meta->build_hash);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]
               ->megaKernel->meta->arg_info);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]
               ->megaKernel->meta);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]
               ->megaKernel->dyn_arguments);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]
               ->megaKernel->data);
      free((char *)command_buffer->fake_single_kernel_cmd_buffers[idx]
               ->megaKernel->name);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]->megaKernel);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]
               ->megaProgram->build_hash);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]
               ->megaProgram->devices);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]->megaProgram);
      free(command_buffer->fake_single_kernel_cmd_buffers[idx]
               ->num_of_settable_arguments);
      idx++;
    }
    free(command_buffer->fake_single_kernel_cmd_buffers);
  } else {
    device->ops->free_kernel(device, command_buffer->megaProgram,
                             command_buffer->megaKernel, 0);
    free(command_buffer->megaKernel->meta->build_hash);
    free(command_buffer->megaKernel->meta->arg_info);
    free(command_buffer->megaKernel->meta);
    free(command_buffer->megaKernel->dyn_arguments);
    free(command_buffer->megaKernel->data);
    free((char *)command_buffer->megaKernel->name);
    free(command_buffer->megaKernel);
    free(command_buffer->megaProgram->build_hash);
    free(command_buffer->megaProgram->devices);
    free(command_buffer->megaProgram);
    free(command_buffer->num_of_settable_arguments);
  }
  delete (AliasingGroups_t *)command_buffer->aliasing_data;
  return CL_SUCCESS;
}

cl_int pocl_mlir_update_mutable_args_of_config(
    const cl_mutable_dispatch_config_khr *cfg,
    cl_command_buffer_khr command_buffer, unsigned num_configs,
    const void **configs) {
  cl_kernel kernel = cfg->command->command.run.kernel;
  // Figure out the first argument index of the command in the megaKernel.
  // This is used for both updating the global size (num_groups) and mutable
  // args.
  _cl_command_node *cmd;
  int cmd_idx = 0;
  LL_FOREACH (command_buffer->cmds, cmd) {
    if (cmd->type != CL_COMMAND_NDRANGE_KERNEL)
      continue;
    if (cmd == cfg->command)
      break;
    cmd_idx++;
  }

  for (int k = 0; k < cfg->num_args; k++) {
    cl_int kernArgStatus =
        POclSetKernelArg(kernel, cfg->arg_list[k].arg_index,
                         cfg->arg_list[k].arg_size, cfg->arg_list[k].arg_value);
    if (kernArgStatus != CL_SUCCESS)
      return kernArgStatus;

    // TODO we can't directly use the cfg->arg_list[k].arg_index as a local
    // index for the argument, because there might be deduplicated buffer args
    // in between, that should be substracted from that. For more explanation
    // see pocl_mlir_cmd_buffer.cc
    int local_idx = cfg->arg_list[k].arg_index;
    if (ARG_IS_LOCAL(kernel->meta->arg_info[local_idx])) {
      POCL_MSG_WARN("Attempting to mutate local arguments, these are currently "
                    "non-mutable, to avoid re-compilation\n");
      // TODO: not spec-compliant
      return CL_INVALID_VALUE;
    }

    AliasingGroups_t *AliasingGroups;
    if (pocl_get_bool_option("POCL_MLIR_DISABLE_CMD_BUFFER_FUSION", 0)) {
      AliasingGroups = (AliasingGroups_t *)command_buffer
                           ->fake_single_kernel_cmd_buffers[cmd_idx]
                           ->aliasing_data;
    } else {
      AliasingGroups = (AliasingGroups_t *)command_buffer->aliasing_data;
    }

    int linear_arg_idx = -1;
    bool found_it = false;
    auto it = AliasingGroups->begin();
    for (; it != AliasingGroups->end(); it++) {
      for (auto it2 = it->begin(); it2 != it->end(); it2++) {
        if (it2->kernel_idx == cmd_idx && it2->local_arg_idx == local_idx) {
          linear_arg_idx = it2->linear_arg_idx;
          found_it = true;
          break;
        }
      }
      if (found_it)
        break;
    }
    if (!found_it) {
      POCL_MSG_WARN("Failed to find the correct kernel idx\n");
      return CL_INVALID_VALUE;
    }
    // Check to make sure that user is attempting to update every single
    // aliasing buffer in the same clUpdateMutables-call. This is not
    // spec-compliant, but it is required for us, since we want to deduplicate
    // the buffers, and avoid recompilation
    if (kernel->meta->arg_info[local_idx].type == POCL_ARG_TYPE_POINTER) {
      for (auto it2 = it->begin(); it2 != it->end(); it2++) {
        int arg_idx_to_find = it2->local_arg_idx;
        int cmd_idx_to_find = it2->kernel_idx;
        bool found_matching_cfg_arg_setting = false;
        for (cl_uint i = 0; i < num_configs; ++i) {
          const cl_mutable_dispatch_config_khr *cfg_to_check =
              (const cl_mutable_dispatch_config_khr *)configs[i];
          _cl_command_node *cmd_tmp;
          int cfg_cmd_idx = 0;
          LL_FOREACH (command_buffer->cmds, cmd_tmp) {
            if (cmd_tmp->type != CL_COMMAND_NDRANGE_KERNEL)
              continue;
            if (cmd_tmp == cfg->command)
              break;
            cfg_cmd_idx++;
          }
          if (cfg_cmd_idx != cmd_idx_to_find)
            continue;

          for (int k = 0; k < cfg_to_check->num_args; k++) {
            if (cfg_to_check->arg_list[k].arg_index == arg_idx_to_find) {
              found_matching_cfg_arg_setting = true;
              break;
            }
          }
          if (found_matching_cfg_arg_setting)
            break;
        }
        if (!found_matching_cfg_arg_setting) {
          POCL_MSG_WARN(
              "Not setting all the aliasing buffer arguments in the same call. "
              "This is required by our implementation, making this part of the "
              "implementation not spec-compliant\n");
          return CL_INVALID_VALUE;
        }
      }
    }

    //    int local_idx_orig = local_idx; // Save this to use as the loop upper
    //    bound, since we are mutating its value later for (int arg_idx = 0;
    //    arg_idx < local_idx_orig; arg_idx++) {
    //      struct pocl_argument_info* tmp_arg_info =
    //      &command_buffer->megaKernel->meta->arg_info[first_arg_index_of_cmd_in_megaKernel
    //      + arg_idx]; if (ARGP_IS_LOCAL(tmp_arg_info)) {
    //        local_idx--;
    //        continue;
    //      }
    //      // Figure out if some of the original pointer argument of the kernel
    //      has been deduplicated int isDeduplicatedArgPointer =
    //      tmp_arg_info->type == POCL_ARG_TYPE_POINTER; int
    //      isOrigKernelArgPointer = kernel->meta->arg_info[arg_idx].type ==
    //      POCL_ARG_TYPE_POINTER; if (isOrigKernelArgPointer &&
    //      !isDeduplicatedArgPointer) {
    //        local_idx--;
    //      }
    //    }
    if (pocl_get_bool_option("POCL_MLIR_DISABLE_CMD_BUFFER_FUSION", 0)) {
      kernArgStatus = POclSetKernelArg(
          command_buffer->fake_single_kernel_cmd_buffers[cmd_idx]->megaKernel,
          linear_arg_idx, cfg->arg_list[k].arg_size,
          cfg->arg_list[k].arg_value);
    } else {
      kernArgStatus = POclSetKernelArg(
          command_buffer->megaKernel, linear_arg_idx, cfg->arg_list[k].arg_size,
          cfg->arg_list[k].arg_value);
    }
    if (kernArgStatus != CL_SUCCESS)
      return kernArgStatus;
  }
  return CL_SUCCESS;
}
