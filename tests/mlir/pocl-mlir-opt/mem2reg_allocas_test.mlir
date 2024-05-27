// RUN: pocl-mlir-opt --pocl-mem2reg --allow-unregistered-dialect --split-input-file %s | %FileCheck %s

module attributes {cir.cl.version = #cir.cl.version<1, 2>, cir.lang = #cir.lang<opencl_c>, cir.opt_info = #cir.opt_info<level = 2, size = 0>, cir.sob = #cir.signed_overflow_behavior<undefined>, cir.triple = "spirv64", dlti.dl_spec = #dlti.dl_spec<i1 = dense<8> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, "dlti.endianness" = "little", "dlti.global_memory_space" = 1 : ui64>, gpu.WGDynamicLocalSize = false, gpu.WGLocalSizeX = 8 : i64, gpu.WGLocalSizeY = 8 : i64, gpu.WGLocalSizeZ = 1 : i64} {
  func.func @mm2_kernel1(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: f32, %arg8: f32) attributes {gpu.kernel} {
    %cst = arith.constant 0.000000e+00 : f32
    %false = arith.constant false
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() {alignment = 8 : i64} : memref<memref<?xf32>>
    %alloca_0 = memref.alloca() {alignment = 8 : i64} : memref<memref<?xf32>>
    %alloca_1 = memref.alloca() {alignment = 8 : i64} : memref<memref<?xf32>>
    %alloca_2 = memref.alloca() {alignment = 4 : i64} : memref<i32>
    %alloca_3 = memref.alloca() {alignment = 4 : i64} : memref<i32>
    %alloca_4 = memref.alloca() {alignment = 4 : i64} : memref<i32>
    %alloca_5 = memref.alloca() {alignment = 4 : i64} : memref<f32>
    %alloca_6 = memref.alloca() {alignment = 4 : i64} : memref<i32>
    %alloca_7 = memref.alloca() {alignment = 4 : i64} : memref<i32>
    memref.store %arg0, %alloca[] : memref<memref<?xf32>>
    memref.store %arg1, %alloca_0[] : memref<memref<?xf32>>
    memref.store %arg2, %alloca_1[] : memref<memref<?xf32>>
    memref.store %arg3, %alloca_2[] : memref<i32>
    memref.store %arg4, %alloca_3[] : memref<i32>
    memref.store %arg5, %alloca_4[] : memref<i32>
    memref.store %arg7, %alloca_5[] : memref<f32>
    %0 = memref.alloca_scope  -> (i64) {
      %8 = scf.execute_region -> i64 {
        %9 = memref.alloca_scope  -> (i64) {
          %14 = scf.execute_region -> i64 {
            %c0_i64 = arith.constant 0 : i64
            %15 = arith.index_cast %c0_i32 : i32 to index
            %16 = scf.index_switch %15 -> i64 
            case 0 {
              %block_id_x = gpu.block_id  x
              %17 = arith.index_cast %block_id_x : index to i64
              scf.yield %17 : i64
            }
            case 1 {
              %block_id_y = gpu.block_id  y
              %17 = arith.index_cast %block_id_y : index to i64
              scf.yield %17 : i64
            }
            case 2 {
              %block_id_z = gpu.block_id  z
              %17 = arith.index_cast %block_id_z : index to i64
              scf.yield %17 : i64
            }
            default {
              scf.yield %c0_i64 : i64
            }
            scf.yield %16 : i64
          }
          memref.alloca_scope.return %14 : i64
        }
        %10 = memref.alloca_scope  -> (i64) {
          %14 = scf.execute_region -> i64 {
            %c0_i64 = arith.constant 0 : i64
            %15 = arith.index_cast %c0_i32 : i32 to index
            %16 = scf.index_switch %15 -> i64 
            case 0 {
              %block_dim_x = gpu.block_dim  x
              %17 = arith.index_cast %block_dim_x : index to i64
              scf.yield %17 : i64
            }
            case 1 {
              %block_dim_y = gpu.block_dim  y
              %17 = arith.index_cast %block_dim_y : index to i64
              scf.yield %17 : i64
            }
            case 2 {
              %block_dim_z = gpu.block_dim  z
              %17 = arith.index_cast %block_dim_z : index to i64
              scf.yield %17 : i64
            }
            default {
              scf.yield %c0_i64 : i64
            }
            scf.yield %16 : i64
          }
          memref.alloca_scope.return %14 : i64
        }
        %11 = arith.muli %9, %10 : i64
        %12 = memref.alloca_scope  -> (i64) {
          %14 = scf.execute_region -> i64 {
            %c0_i64 = arith.constant 0 : i64
            %15 = arith.index_cast %c0_i32 : i32 to index
            %16 = scf.index_switch %15 -> i64 
            case 0 {
              %thread_id_x = gpu.thread_id  x
              %17 = arith.index_cast %thread_id_x : index to i64
              scf.yield %17 : i64
            }
            case 1 {
              %thread_id_y = gpu.thread_id  y
              %17 = arith.index_cast %thread_id_y : index to i64
              scf.yield %17 : i64
            }
            case 2 {
              %thread_id_z = gpu.thread_id  z
              %17 = arith.index_cast %thread_id_z : index to i64
              scf.yield %17 : i64
            }
            default {
              scf.yield %c0_i64 : i64
            }
            scf.yield %16 : i64
          }
          memref.alloca_scope.return %14 : i64
        }
        %13 = arith.addi %11, %12 : i64
        scf.yield %13 : i64
      }
      memref.alloca_scope.return %8 : i64
    }
    %1 = arith.trunci %0 : i64 to i32
    memref.store %1, %alloca_6[] : memref<i32>
    %2 = memref.alloca_scope  -> (i64) {
      %8 = scf.execute_region -> i64 {
        %9 = memref.alloca_scope  -> (i64) {
          %14 = scf.execute_region -> i64 {
            %c0_i64 = arith.constant 0 : i64
            %15 = arith.index_cast %c1_i32 : i32 to index
            %16 = scf.index_switch %15 -> i64 
            case 0 {
              %block_id_x = gpu.block_id  x
              %17 = arith.index_cast %block_id_x : index to i64
              scf.yield %17 : i64
            }
            case 1 {
              %block_id_y = gpu.block_id  y
              %17 = arith.index_cast %block_id_y : index to i64
              scf.yield %17 : i64
            }
            case 2 {
              %block_id_z = gpu.block_id  z
              %17 = arith.index_cast %block_id_z : index to i64
              scf.yield %17 : i64
            }
            default {
              scf.yield %c0_i64 : i64
            }
            scf.yield %16 : i64
          }
          memref.alloca_scope.return %14 : i64
        }
        %10 = memref.alloca_scope  -> (i64) {
          %14 = scf.execute_region -> i64 {
            %c0_i64 = arith.constant 0 : i64
            %15 = arith.index_cast %c1_i32 : i32 to index
            %16 = scf.index_switch %15 -> i64 
            case 0 {
              %block_dim_x = gpu.block_dim  x
              %17 = arith.index_cast %block_dim_x : index to i64
              scf.yield %17 : i64
            }
            case 1 {
              %block_dim_y = gpu.block_dim  y
              %17 = arith.index_cast %block_dim_y : index to i64
              scf.yield %17 : i64
            }
            case 2 {
              %block_dim_z = gpu.block_dim  z
              %17 = arith.index_cast %block_dim_z : index to i64
              scf.yield %17 : i64
            }
            default {
              scf.yield %c0_i64 : i64
            }
            scf.yield %16 : i64
          }
          memref.alloca_scope.return %14 : i64
        }
        %11 = arith.muli %9, %10 : i64
        %12 = memref.alloca_scope  -> (i64) {
          %14 = scf.execute_region -> i64 {
            %c0_i64 = arith.constant 0 : i64
            %15 = arith.index_cast %c1_i32 : i32 to index
            %16 = scf.index_switch %15 -> i64 
            case 0 {
              %thread_id_x = gpu.thread_id  x
              %17 = arith.index_cast %thread_id_x : index to i64
              scf.yield %17 : i64
            }
            case 1 {
              %thread_id_y = gpu.thread_id  y
              %17 = arith.index_cast %thread_id_y : index to i64
              scf.yield %17 : i64
            }
            case 2 {
              %thread_id_z = gpu.thread_id  z
              %17 = arith.index_cast %thread_id_z : index to i64
              scf.yield %17 : i64
            }
            default {
              scf.yield %c0_i64 : i64
            }
            scf.yield %16 : i64
          }
          memref.alloca_scope.return %14 : i64
        }
        %13 = arith.addi %11, %12 : i64
        scf.yield %13 : i64
      }
      memref.alloca_scope.return %8 : i64
    }
    %3 = arith.trunci %2 : i64 to i32
    memref.store %3, %alloca_7[] : memref<i32>
    %4 = memref.load %alloca_7[] : memref<i32>
    %5 = memref.load %alloca_2[] : memref<i32>
    %6 = arith.cmpi slt, %4, %5 : i32
    %7 = scf.if %6 -> (i1) {
      %8 = memref.load %alloca_6[] : memref<i32>
      %9 = memref.load %alloca_3[] : memref<i32>
      %10 = arith.cmpi slt, %8, %9 : i32
      scf.yield %10 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %7 {
      %8 = memref.load %alloca[] : memref<memref<?xf32>>
      %9 = memref.load %alloca_7[] : memref<i32>
      %10 = memref.load %alloca_3[] : memref<i32>
      %11 = arith.muli %9, %10 : i32
      %12 = memref.load %alloca_6[] : memref<i32>
      %13 = arith.addi %11, %12 : i32
      %14 = arith.index_cast %13 : i32 to index
      memref.store %cst, %8[%14] : memref<?xf32>
      %15 = memref.load %alloca_4[] : memref<i32>
      scf.for %arg9 = %c0_i32 to %15 step %c1_i32  : i32 {
        %16 = memref.load %alloca_5[] : memref<f32>
        %17 = memref.load %alloca_0[] : memref<memref<?xf32>>
        %18 = memref.load %alloca_7[] : memref<i32>
        %19 = memref.load %alloca_4[] : memref<i32>
        %20 = arith.muli %18, %19 : i32
        %21 = arith.addi %20, %arg9 : i32
        %22 = arith.index_cast %21 : i32 to index
        %23 = memref.load %17[%22] : memref<?xf32>
        %24 = arith.mulf %16, %23 : f32
        %25 = memref.load %alloca_1[] : memref<memref<?xf32>>
        %26 = memref.load %alloca_3[] : memref<i32>
        %27 = arith.muli %arg9, %26 : i32
        %28 = memref.load %alloca_6[] : memref<i32>
        %29 = arith.addi %27, %28 : i32
        %30 = arith.index_cast %29 : i32 to index
        %31 = memref.load %25[%30] : memref<?xf32>
        %32 = arith.mulf %24, %31 : f32
        %33 = memref.load %alloca[] : memref<memref<?xf32>>
        %34 = arith.muli %18, %26 : i32
        %35 = arith.addi %34, %28 : i32
        %36 = arith.index_cast %35 : i32 to index
        %37 = memref.load %33[%36] : memref<?xf32>
        %38 = arith.addf %37, %32 : f32
        memref.store %38, %33[%36] : memref<?xf32>
      }
    }
    return
  }
}

//			CHECK:	func.func @mm2_kernel1(%[[arg0:.+]]: memref<?xf32>, %[[arg1:.+]]: memref<?xf32>, %[[arg2:.+]]: memref<?xf32>, %[[arg3:.+]]: i32, %[[arg4:.+]]: i32, %[[arg5:.+]]: i32, %[[arg6:.+]]: i32, %[[arg7:.+]]: f32, %[[arg8:.+]]: f32) attributes {gpu.kernel} {
// CHECK-NEXT:    %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:    %[[false:.+]] = arith.constant false
// CHECK-NEXT:    %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[x0:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:      %[[x6:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:        %[[x7:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:          %[[x12:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:            %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:            %[[x13:.+]] = arith.index_cast %[[c0_i32]] : i32 to index
// CHECK-NEXT:            %[[x14:.+]] = scf.index_switch %[[x13]] -> i64 
// CHECK-NEXT:            case 0 {
// CHECK-NEXT:              %[[block_id_x:.+]] = gpu.block_id  x
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_id_x]] : index to i64
// CHECK-NEXT:              scf.yield %15 : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 1 {
// CHECK-NEXT:              %[[block_id_y:.+]] = gpu.block_id  y
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_id_y]] : index to i64
// CHECK-NEXT:              scf.yield %15 : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 2 {
// CHECK-NEXT:              %[[block_id_z:.+]] = gpu.block_id  z
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_id_z]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            default {
// CHECK-NEXT:              scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[x14]] : i64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope.return %[[x12]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[x8:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:          %[[x12:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:            %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:            %[[x13:.+]] = arith.index_cast %[[c0_i32]] : i32 to index
// CHECK-NEXT:            %[[x14:.+]] = scf.index_switch %[[x13]] -> i64 
// CHECK-NEXT:            case 0 {
// CHECK-NEXT:              %[[block_dim_x:.+]] = gpu.block_dim  x
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_dim_x]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 1 {
// CHECK-NEXT:              %[[block_dim_y:.+]] = gpu.block_dim  y
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_dim_y]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 2 {
// CHECK-NEXT:              %[[block_dim_z:.+]] = gpu.block_dim  z
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_dim_z]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            default {
// CHECK-NEXT:              scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[x14]] : i64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope.return %[[x12]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[x9:.+]] = arith.muli %[[x7]], %[[x8]] : i64
// CHECK-NEXT:        %[[x10:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:          %[[x12:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:            %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:            %[[x13:.+]] = arith.index_cast %[[c0_i32]] : i32 to index
// CHECK-NEXT:            %[[x14:.+]] = scf.index_switch %[[x13]] -> i64 
// CHECK-NEXT:            case 0 {
// CHECK-NEXT:              %[[thread_id_x:.+]] = gpu.thread_id  x
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[thread_id_x]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 1 {
// CHECK-NEXT:              %[[thread_id_y:.+]] = gpu.thread_id  y
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[thread_id_y]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 2 {
// CHECK-NEXT:              %[[thread_id_z:.+]] = gpu.thread_id  z
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[thread_id_z]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            default {
// CHECK-NEXT:              scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[x14]] : i64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope.return %[[x12]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[x11:.+]] = arith.addi %[[x9]], %[[x10]] : i64
// CHECK-NEXT:        scf.yield %[[x11]] : i64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.alloca_scope.return %[[x6]] : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[x1:.+]] = arith.trunci %[[x0]] : i64 to i32
// CHECK-NEXT:    %[[x2:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:      %[[x6:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:        %[[x7:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:          %[[x12:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:            %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:            %[[x13:.+]] = arith.index_cast %[[c1_i32]] : i32 to index
// CHECK-NEXT:            %[[x14:.+]] = scf.index_switch %[[x13]] -> i64 
// CHECK-NEXT:            case 0 {
// CHECK-NEXT:              %[[block_id_x:.+]] = gpu.block_id  x
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_id_x]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 1 {
// CHECK-NEXT:              %[[block_id_y:.+]] = gpu.block_id  y
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_id_y]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 2 {
// CHECK-NEXT:              %[[block_id_z:.+]] = gpu.block_id  z
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_id_z]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            default {
// CHECK-NEXT:              scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[x14]] : i64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope.return %[[x12]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[x8:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:          %[[x12:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:            %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:            %[[x13:.+]] = arith.index_cast %[[c1_i32]] : i32 to index
// CHECK-NEXT:            %[[x14:.+]] = scf.index_switch %[[x13]] -> i64 
// CHECK-NEXT:            case 0 {
// CHECK-NEXT:              %[[block_dim_x:.+]] = gpu.block_dim  x
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_dim_x]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 1 {
// CHECK-NEXT:              %[[block_dim_y:.+]] = gpu.block_dim  y
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_dim_y]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 2 {
// CHECK-NEXT:              %[[block_dim_z:.+]] = gpu.block_dim  z
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[block_dim_z]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            default {
// CHECK-NEXT:              scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[x14]] : i64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope.return %[[x12]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[x9:.+]] = arith.muli %[[x7]], %[[x8]] : i64
// CHECK-NEXT:        %[[x10:.+]] = memref.alloca_scope  -> (i64) {
// CHECK-NEXT:          %[[x12:.+]] = scf.execute_region -> i64 {
// CHECK-NEXT:            %[[c0_i64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:            %[[x13:.+]] = arith.index_cast %[[c1_i32]] : i32 to index
// CHECK-NEXT:            %[[x14:.+]] = scf.index_switch %[[x13]] -> i64 
// CHECK-NEXT:            case 0 {
// CHECK-NEXT:              %[[thread_id_x:.+]] = gpu.thread_id  x
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[thread_id_x]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 1 {
// CHECK-NEXT:              %[[thread_id_y:.+]] = gpu.thread_id  y
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[thread_id_y]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            case 2 {
// CHECK-NEXT:              %[[thread_id_z:.+]] = gpu.thread_id  z
// CHECK-NEXT:              %[[x15:.+]] = arith.index_cast %[[thread_id_z]] : index to i64
// CHECK-NEXT:              scf.yield %[[x15]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            default {
// CHECK-NEXT:              scf.yield %[[c0_i64]] : i64
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[x14]] : i64
// CHECK-NEXT:          }
// CHECK-NEXT:          memref.alloca_scope.return %[[x12]] : i64
// CHECK-NEXT:        }
// CHECK-NEXT:        %[[x11:.+]] = arith.addi %[[x9]], %[[x10]] : i64
// CHECK-NEXT:        scf.yield %[[x11]] : i64
// CHECK-NEXT:      }
// CHECK-NEXT:      memref.alloca_scope.return %[[x6]] : i64
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[x3:.+]] = arith.trunci %[[x2]] : i64 to i32
// CHECK-NEXT:    %[[x4:.+]] = arith.cmpi slt, %[[x3]], %[[arg3]] : i32
// CHECK-NEXT:    %[[x5:.+]] = scf.if %[[x4]] -> (i1) {
// CHECK-NEXT:      %[[x6:.+]] = arith.cmpi slt, %[[x1]], %[[arg4]] : i32
// CHECK-NEXT:      scf.yield %[[x6]] : i1
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %[[false]] : i1
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.if %[[x5]] {
// CHECK-NEXT:      %[[x6:.+]] = arith.muli %[[x3]], %[[arg4]] : i32
// CHECK-NEXT:      %[[x7:.+]] = arith.addi %[[x6]], %[[x1]] : i32
// CHECK-NEXT:      %[[x8:.+]] = arith.index_cast %[[x7]] : i32 to index
// CHECK-NEXT:      memref.store %[[cst]], %[[arg0]][%[[x8]]] : memref<?xf32>
// CHECK-NEXT:      scf.for %[[arg9:.+]] = %[[c0_i32]] to %[[arg5]] step %[[c1_i32]]  : i32 {
// CHECK-NEXT:        %[[x9:.+]] = arith.muli %[[x3]], %[[arg5]] : i32
// CHECK-NEXT:        %[[x10:.+]] = arith.addi %[[x9]], %[[arg9]] : i32
// CHECK-NEXT:        %[[x11:.+]] = arith.index_cast %[[x10]] : i32 to index
// CHECK-NEXT:        %[[x12:.+]] = memref.load %[[arg1]][%[[x11]]] : memref<?xf32>
// CHECK-NEXT:        %[[x13:.+]] = arith.mulf %[[arg7]], %[[x12]] : f32
// CHECK-NEXT:        %[[x14:.+]] = arith.muli %[[arg9]], %[[arg4]] : i32
// CHECK-NEXT:        %[[x15:.+]] = arith.addi %[[x14]], %[[x1]] : i32
// CHECK-NEXT:        %[[x16:.+]] = arith.index_cast %[[x15]] : i32 to index
// CHECK-NEXT:        %[[x17:.+]] = memref.load %[[arg2]][%[[x16]]] : memref<?xf32>
// CHECK-NEXT:        %[[x18:.+]] = arith.mulf %[[x13]], %[[x17]] : f32
// CHECK-NEXT:        %[[x19:.+]] = arith.muli %[[x3]], %[[arg4]] : i32
// CHECK-NEXT:        %[[x20:.+]] = arith.addi %[[x19]], %[[x1]] : i32
// CHECK-NEXT:        %[[x21:.+]] = arith.index_cast %[[x20]] : i32 to index
// CHECK-NEXT:        %[[x22:.+]] = memref.load %[[arg0]][%[[x21]]] : memref<?xf32>
// CHECK-NEXT:        %[[x23:.+]] = arith.addf %[[x22]], %[[x18]] : f32
// CHECK-NEXT:        memref.store %[[x23]], %[[arg0]][%[[x21]]] : memref<?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
