module  {
   AIE.device(ipu){

   %tile00 = AIE.tile(0, 0)
   %tile02 = AIE.tile(0, 2)
   AIE.objectFifo @itbuffer_0___ITout___mmul_0___pA(%tile00, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<512xi32>>
   AIE.objectFifo @itbuffer_1___ITout___mmul_0___pB(%tile00, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<2048xi32>>
   AIE.objectFifo @mmul_0___pC___itbuffer_2___ITin(%tile02, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<256xi32>>


   func.func private @mmul(memref<512xi32>, memref<2048xi32>, memref<256xi32>) -> ()

   AIE.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview1 = AIE.objectFifo.acquire @itbuffer_0___ITout___mmul_0___pA(Consume, 1) : !AIE.objectFifoSubview<memref<512xi32>>
         %elem1 = AIE.objectFifo.subview.access %subview1[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
         %subview2 = AIE.objectFifo.acquire @itbuffer_1___ITout___mmul_0___pB(Consume, 1) : !AIE.objectFifoSubview<memref<2048xi32>>
         %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<2048xi32>> -> memref<2048xi32>
         %subview3 = AIE.objectFifo.acquire @mmul_0___pC___itbuffer_2___ITin(Produce, 1) : !AIE.objectFifoSubview<memref<256xi32>>
         %elem3 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<256xi32>> -> memref<256xi32>

         func.call @mmul(%elem1, %elem2, %elem3) : (memref<512xi32>, memref<2048xi32>, memref<256xi32>) -> ()

         AIE.objectFifo.release @itbuffer_0___ITout___mmul_0___pA(Consume, 1)
         AIE.objectFifo.release @itbuffer_1___ITout___mmul_0___pB(Consume, 1)
         AIE.objectFifo.release @mmul_0___pC___itbuffer_2___ITin(Produce, 1)
      }
      AIE.end
   } { link_with="mmul.o" }

func.func @sequence(%itbuffer_0 : memref<8x64xi32>,%itbuffer_1 : memref<128x16xi32>,%itbuffer_2 : memref<8x32xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4096 = arith.constant 4096 : i32
    %c16 = arith.constant 16 : i32
    %c64 = arith.constant 64 : i32
    %c8192 = arith.constant 8192 : i32
    %c32 = arith.constant 32 : i32


    %c8 = arith.constant 8 : i32
    %c128 = arith.constant 128 : i32
    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_2[%c0, %c0, %c0, %c0][%c1, %c1, %c8, %c32][%c0, %c0, %c0]){ metadata= @mmul_0___pC___itbuffer_2___ITin, id = 2 : i32 } :(i32, i32, memref<8x32xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c0][%c1, %c1, %c8, %c64][%c0, %c0, %c0]){ metadata= @itbuffer_0___ITout___mmul_0___pA, id = 0 : i32 } :(i32, i32, memref<8x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c0][%c1, %c1, %c128, %c16][%c0, %c0, %c0]){ metadata= @itbuffer_1___ITout___mmul_0___pB, id = 1 : i32 } :(i32, i32, memref<128x16xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.sync {column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    return
}
 }
}
