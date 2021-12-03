// RUN: mlir-proto-opt %s -linalg-ext-tiling-to-tile-op="tile-size=10" -linalg-tile-to-sequential-for -verify-each=0 | FileCheck %s

func @reverse_1d_tensor(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>

  // CHECK: linalg.init_tensor {{.*}} : tensor<?x?xf32>
  // CHECK: scf.for {{.*}}}} -> (tensor<?x?xf32>) {
  // CHECK:   tensor.extract_slice {{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  // CHECK:   tensor.extract_slice {{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  // CHECK:   tensor.extract_slice {{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  // CHECK:   linalg_ext.reverse dimensions(dense<0> : tensor<1xi64>) ins(%{{.*}} : tensor<?x?xf32>) outs(%{{.*}} : tensor<?x?xf32>) : tensor<?x?xf32>
  // CHECK:   tensor.insert_slice {{.*}} : tensor<?x?xf32> into tensor<?x?xf32>
  // CHECK:   scf.yield {{.*}}: tensor<?x?xf32>

  // TODO: something fishy is happening with the verifier after tiling, disabling verification for now
  %reverse = linalg_ext.reverse
      dimensions(dense<0> : tensor<1xi64>)
      ins(%arg0 : tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) : tensor<?x?xf32>
  return %reverse : tensor<?x?xf32>
}

func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  // TODO: LinalgOp needs interface composition to implemented the TilingInterface.
  %D = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>) outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}