module  {
   AIE.device(ipu){

   %tile00 = AIE.tile(0, 0)
   %tile01 = AIE.tile(0, 1)
   %tile02 = AIE.tile(0, 2)
   %tile03 = AIE.tile(0, 3)
   %tile04 = AIE.tile(0, 4)
   %tile05 = AIE.tile(0, 5)
   AIE.objectFifo @itbuffer_0___ITout___mtbuffer_0___MTin(%tile00, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<512xi32>>
   AIE.objectFifo @itbuffer_1___ITout___mtbuffer_1___MTin(%tile00, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<2048xi32>>
   AIE.objectFifo @mtbuffer_1___MTout___mmul_start_0___pB(%tile01, {%tile05}, 2 : i32) : !AIE.objectFifo<memref<512xi32>>
   AIE.objectFifo @mtbuffer_1___MTout___mmul_start_1___pB(%tile01, {%tile04}, 2 : i32) : !AIE.objectFifo<memref<512xi32>>
   AIE.objectFifo @mtbuffer_1___MTout___mmul_start_2___pB(%tile01, {%tile03}, 2 : i32) : !AIE.objectFifo<memref<512xi32>>
   AIE.objectFifo @mtbuffer_1___MTout___mmul_start_3___pB(%tile01, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<512xi32>>
   AIE.objectFifo @mmul_start_0___pC___mtbuffer_2___MTin(%tile05, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mmul_start_1___pC___mtbuffer_2___MTin(%tile04, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mmul_start_2___pC___mtbuffer_2___MTin(%tile03, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mmul_start_3___pC___mtbuffer_2___MTin(%tile02, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_2___MTout___itbuffer_2___ITin(%tile01, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mtbuffer_0__MTout(%tile01, {%tile05, %tile03, %tile02, %tile04}, [2,2,2,2,2]) : !AIE.objectFifo<memref<512xi32>>

   AIE.objectFifo.link [@itbuffer_0___ITout___mtbuffer_0___MTin ] -> [@mtbuffer_0__MTout] ()
   AIE.objectFifo.link [@itbuffer_1___ITout___mtbuffer_1___MTin ] -> [@mtbuffer_1___MTout___mmul_start_0___pB,@mtbuffer_1___MTout___mmul_start_1___pB,@mtbuffer_1___MTout___mmul_start_2___pB,@mtbuffer_1___MTout___mmul_start_3___pB] ()
   AIE.objectFifo.link [@mmul_start_0___pC___mtbuffer_2___MTin,@mmul_start_1___pC___mtbuffer_2___MTin,@mmul_start_2___pC___mtbuffer_2___MTin,@mmul_start_3___pC___mtbuffer_2___MTin ] -> [@mtbuffer_2___MTout___itbuffer_2___ITin] ()

   func.func private @mmul_start(memref<512xi32>, memref<512xi32>, memref<1024xi32>) -> ()

   AIE.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview5 = AIE.objectFifo.acquire @mtbuffer_0__MTout(Consume, 1) : !AIE.objectFifoSubview<memref<512xi32>>
         %elem5 = AIE.objectFifo.subview.access %subview5[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
         %subview11 = AIE.objectFifo.acquire @mtbuffer_1___MTout___mmul_start_3___pB(Consume, 1) : !AIE.objectFifoSubview<memref<512xi32>>
         %elem11 = AIE.objectFifo.subview.access %subview11[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
         %subview9 = AIE.objectFifo.acquire @mmul_start_3___pC___mtbuffer_2___MTin(Produce, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem9 = AIE.objectFifo.subview.access %subview9[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>

         func.call @mmul_start(%elem5, %elem11, %elem9) : (memref<512xi32>, memref<512xi32>, memref<1024xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout___mmul_start_3___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_3___pC___mtbuffer_2___MTin(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_start.o" }

   AIE.core(%tile03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview4 = AIE.objectFifo.acquire @mtbuffer_0__MTout(Consume, 1) : !AIE.objectFifoSubview<memref<512xi32>>
         %elem4 = AIE.objectFifo.subview.access %subview4[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
         %subview11 = AIE.objectFifo.acquire @mtbuffer_1___MTout___mmul_start_2___pB(Consume, 1) : !AIE.objectFifoSubview<memref<512xi32>>
         %elem11 = AIE.objectFifo.subview.access %subview11[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
         %subview8 = AIE.objectFifo.acquire @mmul_start_2___pC___mtbuffer_2___MTin(Produce, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem8 = AIE.objectFifo.subview.access %subview8[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>

         func.call @mmul_start(%elem4, %elem11, %elem8) : (memref<512xi32>, memref<512xi32>, memref<1024xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout___mmul_start_2___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_2___pC___mtbuffer_2___MTin(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_start.o" }

   AIE.core(%tile04) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview3 = AIE.objectFifo.acquire @mtbuffer_0__MTout(Consume, 1) : !AIE.objectFifoSubview<memref<512xi32>>
         %elem3 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
         %subview11 = AIE.objectFifo.acquire @mtbuffer_1___MTout___mmul_start_1___pB(Consume, 1) : !AIE.objectFifoSubview<memref<512xi32>>
         %elem11 = AIE.objectFifo.subview.access %subview11[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
         %subview7 = AIE.objectFifo.acquire @mmul_start_1___pC___mtbuffer_2___MTin(Produce, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem7 = AIE.objectFifo.subview.access %subview7[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>

         func.call @mmul_start(%elem3, %elem11, %elem7) : (memref<512xi32>, memref<512xi32>, memref<1024xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout___mmul_start_1___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_1___pC___mtbuffer_2___MTin(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_start.o" }

   AIE.core(%tile05) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview2 = AIE.objectFifo.acquire @mtbuffer_0__MTout(Consume, 1) : !AIE.objectFifoSubview<memref<512xi32>>
         %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
         %subview11 = AIE.objectFifo.acquire @mtbuffer_1___MTout___mmul_start_0___pB(Consume, 1) : !AIE.objectFifoSubview<memref<512xi32>>
         %elem11 = AIE.objectFifo.subview.access %subview11[0] : !AIE.objectFifoSubview<memref<512xi32>> -> memref<512xi32>
         %subview6 = AIE.objectFifo.acquire @mmul_start_0___pC___mtbuffer_2___MTin(Produce, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem6 = AIE.objectFifo.subview.access %subview6[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>

         func.call @mmul_start(%elem2, %elem11, %elem6) : (memref<512xi32>, memref<512xi32>, memref<1024xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout___mmul_start_0___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_0___pC___mtbuffer_2___MTin(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_start.o" }

func.func @sequence(%itbuffer_0 : memref<32x16xi32>,%itbuffer_1 : memref<32x64xi32>,%itbuffer_2 : memref<32x128xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c16384 = arith.constant 16384 : i32
    %c64 = arith.constant 64 : i32
    %c16 = arith.constant 16 : i32
    %c256 = arith.constant 256 : i32
    %c4096 = arith.constant 4096 : i32
    %c4 = arith.constant 4 : i32
    %c32768 = arith.constant 32768 : i32
    %c32 = arith.constant 32 : i32


    %c128 = arith.constant 128 : i32
    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_2[%c0, %c0, %c0, %c0][%c1, %c1, %c32, %c128][%c0, %c0, %c0]){ metadata= @mtbuffer_2___MTout___itbuffer_2___ITin, id = 2 : i32 } :(i32, i32, memref<32x128xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c0][%c1, %c1, %c32, %c16][%c0, %c0, %c0]){ metadata= @itbuffer_0___ITout___mtbuffer_0___MTin, id = 1 : i32 } :(i32, i32, memref<32x16xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c0][%c1, %c1, %c32, %c64][%c0, %c0, %c0]){ metadata= @itbuffer_1___ITout___mtbuffer_1___MTin, id = 0 : i32 } :(i32, i32, memref<32x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.sync {column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    return
}
 }
}
