/* PoclOps.td

   Copyright (c) 2025 Topi Leppanen

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

#ifndef POCL_OPS
#define POCL_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def PoclDialect : Dialect {
  let name = "pocl";
  let cppNamespace = "::mlir::pocl";
}

class PoclDialect_Op<string mnemonic, list<Trait> traits = []> :
  Op<PoclDialect, mnemonic, traits>;

def WorkDimOp
    : PoclDialect_Op<"work_dim", []> {

  let results = (outs I32:$output);
  let summary = "Return value of OpenCL get_work_dim()";
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def CacheLoad
    : PoclDialect_Op<"cacheload"> {

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);
  let builders = [
    OpBuilder<(ins "Value":$memref, CArg<"ValueRange", "{}">:$indices), [{
      auto memrefType = cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.types.push_back(memrefType.getElementType());
    }]>];
  let summary = "load from cross barrier cache";
  let description = [{}];
}
//===----------------------------------------------------------------------===//
// SubIndexOp
//===----------------------------------------------------------------------===//

def SubIndexOp : PoclDialect_Op<"subindex", [
  ViewLikeOpInterface, Pure
]> {
  let summary = "memref subview operation";

  let arguments = (ins AnyMemRef : $source, Index : $index);
  let results = (outs AnyMemRef : $result);

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    ::mlir::Value getViewSource() { return getSource(); }
  }];
}
// HasParent<"ParallelOp">,
def BarrierOp
    : PoclDialect_Op<"barrier",
                   [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {

  let arguments = (ins Variadic<Index>:$indices);
  let summary = "barrier for parallel loops";
  let description = [{}];
  let hasCanonicalizer = true;
}


#endif
