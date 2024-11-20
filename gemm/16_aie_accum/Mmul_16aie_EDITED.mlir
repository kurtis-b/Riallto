module  {
   AIE.device(ipu){

   %tile00 = AIE.tile(0, 0)
   %tile01 = AIE.tile(0, 1)
   %tile02 = AIE.tile(0, 2)
   %tile03 = AIE.tile(0, 3)
   %tile04 = AIE.tile(0, 4)
   %tile05 = AIE.tile(0, 5)
   %tile10 = AIE.tile(1, 0)
   %tile11 = AIE.tile(1, 1)
   %tile12 = AIE.tile(1, 2)
   %tile13 = AIE.tile(1, 3)
   %tile14 = AIE.tile(1, 4)
   %tile15 = AIE.tile(1, 5)
   %tile20 = AIE.tile(2, 0)
   %tile21 = AIE.tile(2, 1)
   %tile22 = AIE.tile(2, 2)
   %tile23 = AIE.tile(2, 3)
   %tile24 = AIE.tile(2, 4)
   %tile25 = AIE.tile(2, 5)
   %tile30 = AIE.tile(3, 0)
   %tile31 = AIE.tile(3, 1)
   %tile32 = AIE.tile(3, 2)
   %tile33 = AIE.tile(3, 3)
   %tile34 = AIE.tile(3, 4)
   %tile35 = AIE.tile(3, 5)

   // Broadcast 4 submatrices of A to a row of AIEs
   AIE.objectFifo @itbuffer_0___ITout_00___mtbuffer_0___MTin_01(%tile00, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @itbuffer_0___ITout_10___mtbuffer_0___MTin_11(%tile10, {%tile11}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @itbuffer_0___ITout_20___mtbuffer_0___MTin_21(%tile20, {%tile21}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @itbuffer_0___ITout_30___mtbuffer_0___MTin_31(%tile30, {%tile31}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_0__MTout_01(%tile01, {%tile05, %tile15, %tile25, %tile35}, [2,2,2,2,2]) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_0__MTout_11(%tile11, {%tile04, %tile14, %tile24, %tile34}, [2,2,2,2,2]) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_0__MTout_21(%tile21, {%tile03, %tile13, %tile23, %tile33}, [2,2,2,2,2]) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_0__MTout_31(%tile31, {%tile02, %tile12, %tile22, %tile32}, [2,2,2,2,2]) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo.link [@itbuffer_0___ITout_00___mtbuffer_0___MTin_01 ] -> [@mtbuffer_0__MTout_01] ()
   AIE.objectFifo.link [@itbuffer_0___ITout_10___mtbuffer_0___MTin_11 ] -> [@mtbuffer_0__MTout_11] ()
   AIE.objectFifo.link [@itbuffer_0___ITout_20___mtbuffer_0___MTin_21 ] -> [@mtbuffer_0__MTout_21] ()
   AIE.objectFifo.link [@itbuffer_0___ITout_30___mtbuffer_0___MTin_31 ] -> [@mtbuffer_0__MTout_31] ()
   
   // Packet split different sets of 4 submatices of B to 4 columns of 4 AIEs, i.e. each of the 16 AIEs will get a different submatrix of B
   AIE.objectFifo @itbuffer_1___ITout_00___mtbuffer_1___MTin_01(%tile00, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_01___mmul_start_05___pB(%tile01, {%tile05}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_01___mmul_accum_04___pB(%tile01, {%tile04}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_01___mmul_accum_03___pB(%tile01, {%tile03}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_01___mmul_end_02___pB(%tile01, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo.link [@itbuffer_1___ITout_00___mtbuffer_1___MTin_01 ] -> [@mtbuffer_1___MTout_01___mmul_start_05___pB, @mtbuffer_1___MTout_01___mmul_accum_04___pB,@mtbuffer_1___MTout_01___mmul_accum_03___pB,@mtbuffer_1___MTout_01___mmul_end_02___pB] ()

   AIE.objectFifo @itbuffer_1___ITout_10___mtbuffer_1___MTin_11(%tile10, {%tile11}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_11___mmul_start_15___pB(%tile11, {%tile15}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_11___mmul_accum_14___pB(%tile11, {%tile14}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_11___mmul_accum_13___pB(%tile11, {%tile13}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_11___mmul_end_12___pB(%tile11, {%tile12}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo.link [@itbuffer_1___ITout_10___mtbuffer_1___MTin_11 ] -> [@mtbuffer_1___MTout_11___mmul_start_15___pB, @mtbuffer_1___MTout_11___mmul_accum_14___pB,@mtbuffer_1___MTout_11___mmul_accum_13___pB,@mtbuffer_1___MTout_11___mmul_end_12___pB] ()

   AIE.objectFifo @itbuffer_1___ITout_20___mtbuffer_1___MTin_21(%tile20, {%tile21}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_21___mmul_start_25___pB(%tile21, {%tile25}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_21___mmul_accum_24___pB(%tile21, {%tile24}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_21___mmul_accum_23___pB(%tile21, {%tile23}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_21___mmul_end_22___pB(%tile21, {%tile22}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo.link [@itbuffer_1___ITout_20___mtbuffer_1___MTin_21 ] -> [@mtbuffer_1___MTout_21___mmul_start_25___pB, @mtbuffer_1___MTout_21___mmul_accum_24___pB,@mtbuffer_1___MTout_21___mmul_accum_23___pB,@mtbuffer_1___MTout_21___mmul_end_22___pB] ()

   AIE.objectFifo @itbuffer_1___ITout_30___mtbuffer_1___MTin_31(%tile30, {%tile31}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_31___mmul_start_35___pB(%tile31, {%tile35}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_31___mmul_accum_34___pB(%tile31, {%tile34}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_31___mmul_accum_33___pB(%tile31, {%tile33}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo @mtbuffer_1___MTout_31___mmul_end_32___pB(%tile31, {%tile32}, 2 : i32) : !AIE.objectFifo<memref<1024xi32>>
   AIE.objectFifo.link [@itbuffer_1___ITout_30___mtbuffer_1___MTin_31 ] -> [@mtbuffer_1___MTout_31___mmul_start_35___pB, @mtbuffer_1___MTout_31___mmul_accum_34___pB,@mtbuffer_1___MTout_31___mmul_accum_33___pB,@mtbuffer_1___MTout_31___mmul_end_32___pB] ()

   // Accumulation from the the 6th row to 5th row, 5th to 4th, 4th to 3rd, 3rd to 2nd, 2nd to 0th row (interface tiles)
   AIE.objectFifo @mmul_start_05__pC(%tile05, {%tile04}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_accum_04__pC(%tile04, {%tile03}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_accum_03__pC(%tile03, {%tile02}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_end_02___pC(%tile02, {%tile01}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mtbuffer_2___MTout___itbuffer_2___ITin_00(%tile01, {%tile00}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo.link [@mmul_end_02___pC ] -> [@mtbuffer_2___MTout___itbuffer_2___ITin_00] ()

   AIE.objectFifo @mmul_start_15__pC(%tile15, {%tile14}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_accum_14__pC(%tile14, {%tile13}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_accum_13__pC(%tile13, {%tile12}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_end_12___pC(%tile12, {%tile11}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mtbuffer_2___MTout___itbuffer_2___ITin_10(%tile11, {%tile10}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo.link [@mmul_end_12___pC ] -> [@mtbuffer_2___MTout___itbuffer_2___ITin_10] ()

   AIE.objectFifo @mmul_start_25__pC(%tile25, {%tile24}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_accum_24__pC(%tile24, {%tile23}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_accum_23__pC(%tile23, {%tile22}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_end_22___pC(%tile22, {%tile21}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mtbuffer_2___MTout___itbuffer_2___ITin_20(%tile21, {%tile20}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo.link [@mmul_end_22___pC ] -> [@mtbuffer_2___MTout___itbuffer_2___ITin_20] ()

   AIE.objectFifo @mmul_start_35__pC(%tile35, {%tile34}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_accum_34__pC(%tile34, {%tile33}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_accum_33__pC(%tile33, {%tile32}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mmul_end_32___pC(%tile32, {%tile31}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo @mtbuffer_2___MTout___itbuffer_2___ITin_30(%tile31, {%tile30}, 2 : i32) : !AIE.objectFifo<memref<4096xi32>>
   AIE.objectFifo.link [@mmul_end_32___pC ] -> [@mtbuffer_2___MTout___itbuffer_2___ITin_30] ()

   func.func private @mmul_start(memref<1024xi32>, memref<1024xi32>, memref<4096xi32>) -> ()
   func.func private @mmul_accum(memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()
   func.func private @mmul_end(memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

   AIE.core(%tile05) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview11 = AIE.objectFifo.acquire @mtbuffer_0__MTout_01(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem11 = AIE.objectFifo.subview.access %subview11[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview5 = AIE.objectFifo.acquire @mtbuffer_1___MTout_01___mmul_start_05___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem5 = AIE.objectFifo.subview.access %subview5[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview9 = AIE.objectFifo.acquire @mmul_start_05__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem9 = AIE.objectFifo.subview.access %subview9[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_start(%elem11, %elem5, %elem9) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_01(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_01___mmul_start_05___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_05__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_start.o" }

   AIE.core(%tile04) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview11 = AIE.objectFifo.acquire @mtbuffer_0__MTout_11(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem11 = AIE.objectFifo.subview.access %subview11[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview4 = AIE.objectFifo.acquire @mtbuffer_1___MTout_01___mmul_accum_04___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem4 = AIE.objectFifo.subview.access %subview4[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview9 = AIE.objectFifo.acquire @mmul_start_05__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem9 = AIE.objectFifo.subview.access %subview9[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview8 = AIE.objectFifo.acquire @mmul_accum_04__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem8 = AIE.objectFifo.subview.access %subview8[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_accum(%elem11, %elem4, %elem9, %elem8) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_11(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_01___mmul_accum_04___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_05__pC(Consume, 1)
         AIE.objectFifo.release @mmul_accum_04__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_accum.o" }

   AIE.core(%tile03) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview11 = AIE.objectFifo.acquire @mtbuffer_0__MTout_21(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem11 = AIE.objectFifo.subview.access %subview11[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview3 = AIE.objectFifo.acquire @mtbuffer_1___MTout_01___mmul_accum_03___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem3 = AIE.objectFifo.subview.access %subview3[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview8 = AIE.objectFifo.acquire @mmul_accum_04__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem8 = AIE.objectFifo.subview.access %subview8[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview7 = AIE.objectFifo.acquire @mmul_accum_03__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem7 = AIE.objectFifo.subview.access %subview7[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_accum(%elem11, %elem3, %elem8, %elem7) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_21(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_01___mmul_accum_03___pB(Consume, 1)
         AIE.objectFifo.release @mmul_accum_04__pC(Consume, 1)
         AIE.objectFifo.release @mmul_accum_03__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_accum.o" }

   AIE.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview11 = AIE.objectFifo.acquire @mtbuffer_0__MTout_31(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem11 = AIE.objectFifo.subview.access %subview11[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview2 = AIE.objectFifo.acquire @mtbuffer_1___MTout_01___mmul_end_02___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem2 = AIE.objectFifo.subview.access %subview2[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview7 = AIE.objectFifo.acquire @mmul_accum_03__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem7 = AIE.objectFifo.subview.access %subview7[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview6 = AIE.objectFifo.acquire @mmul_end_02___pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem6 = AIE.objectFifo.subview.access %subview6[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_end(%elem11, %elem2, %elem7, %elem6) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_31(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_01___mmul_end_02___pB(Consume, 1)
         AIE.objectFifo.release @mmul_accum_03__pC(Consume, 1)
         AIE.objectFifo.release @mmul_end_02___pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_end.o" }

   AIE.core(%tile15) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview16 = AIE.objectFifo.acquire @mtbuffer_0__MTout_01(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem16 = AIE.objectFifo.subview.access %subview16[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview15 = AIE.objectFifo.acquire @mtbuffer_1___MTout_11___mmul_start_15___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem15 = AIE.objectFifo.subview.access %subview15[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview17 = AIE.objectFifo.acquire @mmul_start_15__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem17 = AIE.objectFifo.subview.access %subview17[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_start(%elem16, %elem15, %elem17) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_01(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_11___mmul_start_15___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_15__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_start.o" }

   AIE.core(%tile14) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview16 = AIE.objectFifo.acquire @mtbuffer_0__MTout_11(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem16 = AIE.objectFifo.subview.access %subview16[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview14 = AIE.objectFifo.acquire @mtbuffer_1___MTout_11___mmul_accum_14___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem14 = AIE.objectFifo.subview.access %subview14[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview17 = AIE.objectFifo.acquire @mmul_start_15__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem17 = AIE.objectFifo.subview.access %subview17[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview18 = AIE.objectFifo.acquire @mmul_accum_14__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem18 = AIE.objectFifo.subview.access %subview18[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_accum(%elem16, %elem14, %elem17, %elem18) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_11(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_11___mmul_accum_14___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_15__pC(Consume, 1)
         AIE.objectFifo.release @mmul_accum_14__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_accum.o" }

   AIE.core(%tile13) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview16 = AIE.objectFifo.acquire @mtbuffer_0__MTout_21(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem16 = AIE.objectFifo.subview.access %subview16[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview13 = AIE.objectFifo.acquire @mtbuffer_1___MTout_11___mmul_accum_13___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem13 = AIE.objectFifo.subview.access %subview13[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview18 = AIE.objectFifo.acquire @mmul_accum_14__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem18 = AIE.objectFifo.subview.access %subview18[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview19 = AIE.objectFifo.acquire @mmul_accum_13__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem19 = AIE.objectFifo.subview.access %subview19[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_accum(%elem16, %elem13, %elem18, %elem19) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_21(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_11___mmul_accum_13___pB(Consume, 1)
         AIE.objectFifo.release @mmul_accum_14__pC(Consume, 1)
         AIE.objectFifo.release @mmul_accum_13__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_accum.o" }

   AIE.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview16 = AIE.objectFifo.acquire @mtbuffer_0__MTout_31(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem16 = AIE.objectFifo.subview.access %subview16[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview12 = AIE.objectFifo.acquire @mtbuffer_1___MTout_11___mmul_end_12___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem12 = AIE.objectFifo.subview.access %subview12[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview19 = AIE.objectFifo.acquire @mmul_accum_13__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem19 = AIE.objectFifo.subview.access %subview19[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview20 = AIE.objectFifo.acquire @mmul_end_12___pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem20 = AIE.objectFifo.subview.access %subview20[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_end(%elem16, %elem12, %elem19, %elem20) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_31(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_11___mmul_end_12___pB(Consume, 1)
         AIE.objectFifo.release @mmul_accum_13__pC(Consume, 1)
         AIE.objectFifo.release @mmul_end_12___pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_end.o" }

   AIE.core(%tile25) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview25 = AIE.objectFifo.acquire @mtbuffer_0__MTout_01(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem25 = AIE.objectFifo.subview.access %subview25[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview24 = AIE.objectFifo.acquire @mtbuffer_1___MTout_21___mmul_start_25___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem24 = AIE.objectFifo.subview.access %subview24[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview26 = AIE.objectFifo.acquire @mmul_start_25__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem26 = AIE.objectFifo.subview.access %subview26[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_start(%elem25, %elem24, %elem26) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_01(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_21___mmul_start_25___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_25__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_start.o" }

   AIE.core(%tile24) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview25 = AIE.objectFifo.acquire @mtbuffer_0__MTout_11(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem25 = AIE.objectFifo.subview.access %subview25[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview23 = AIE.objectFifo.acquire @mtbuffer_1___MTout_21___mmul_accum_24___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem23 = AIE.objectFifo.subview.access %subview23[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview26 = AIE.objectFifo.acquire @mmul_start_25__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem26 = AIE.objectFifo.subview.access %subview26[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview27 = AIE.objectFifo.acquire @mmul_accum_24__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem27 = AIE.objectFifo.subview.access %subview27[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_accum(%elem25, %elem23, %elem26, %elem27) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_11(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_21___mmul_accum_24___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_25__pC(Consume, 1)
         AIE.objectFifo.release @mmul_accum_24__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_accum.o" }

   AIE.core(%tile23) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview25 = AIE.objectFifo.acquire @mtbuffer_0__MTout_21(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem25 = AIE.objectFifo.subview.access %subview25[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview22 = AIE.objectFifo.acquire @mtbuffer_1___MTout_21___mmul_accum_23___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem22 = AIE.objectFifo.subview.access %subview22[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview27 = AIE.objectFifo.acquire @mmul_accum_24__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem27 = AIE.objectFifo.subview.access %subview27[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview28 = AIE.objectFifo.acquire @mmul_accum_23__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem28 = AIE.objectFifo.subview.access %subview28[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_accum(%elem25, %elem22, %elem27, %elem28) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_21(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_21___mmul_accum_23___pB(Consume, 1)
         AIE.objectFifo.release @mmul_accum_24__pC(Consume, 1)
         AIE.objectFifo.release @mmul_accum_23__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_accum.o" }

   AIE.core(%tile22) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview25 = AIE.objectFifo.acquire @mtbuffer_0__MTout_31(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem25 = AIE.objectFifo.subview.access %subview25[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview21 = AIE.objectFifo.acquire @mtbuffer_1___MTout_21___mmul_end_22___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem21 = AIE.objectFifo.subview.access %subview21[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview28 = AIE.objectFifo.acquire @mmul_accum_23__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem28 = AIE.objectFifo.subview.access %subview28[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview29 = AIE.objectFifo.acquire @mmul_end_22___pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem29 = AIE.objectFifo.subview.access %subview29[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_end(%elem25, %elem21, %elem28, %elem29) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_31(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_21___mmul_end_22___pB(Consume, 1)
         AIE.objectFifo.release @mmul_accum_23__pC(Consume, 1)
         AIE.objectFifo.release @mmul_end_22___pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_end.o" }

   AIE.core(%tile35) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview34 = AIE.objectFifo.acquire @mtbuffer_0__MTout_01(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem34 = AIE.objectFifo.subview.access %subview34[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview33 = AIE.objectFifo.acquire @mtbuffer_1___MTout_31___mmul_start_35___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem33 = AIE.objectFifo.subview.access %subview33[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview35 = AIE.objectFifo.acquire @mmul_start_35__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem35 = AIE.objectFifo.subview.access %subview35[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_start(%elem34, %elem33, %elem35) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_01(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_31___mmul_start_35___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_35__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_start.o" }

   AIE.core(%tile34) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview34 = AIE.objectFifo.acquire @mtbuffer_0__MTout_11(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem34 = AIE.objectFifo.subview.access %subview34[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview32 = AIE.objectFifo.acquire @mtbuffer_1___MTout_31___mmul_accum_34___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem32 = AIE.objectFifo.subview.access %subview32[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview35 = AIE.objectFifo.acquire @mmul_start_35__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem35 = AIE.objectFifo.subview.access %subview35[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview36 = AIE.objectFifo.acquire @mmul_accum_34__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem36 = AIE.objectFifo.subview.access %subview36[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_accum(%elem34, %elem32, %elem35, %elem36) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_11(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_31___mmul_accum_34___pB(Consume, 1)
         AIE.objectFifo.release @mmul_start_35__pC(Consume, 1)
         AIE.objectFifo.release @mmul_accum_34__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_accum.o" }

   AIE.core(%tile33) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview34 = AIE.objectFifo.acquire @mtbuffer_0__MTout_21(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem34 = AIE.objectFifo.subview.access %subview34[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview31 = AIE.objectFifo.acquire @mtbuffer_1___MTout_31___mmul_accum_33___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem31 = AIE.objectFifo.subview.access %subview31[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview36 = AIE.objectFifo.acquire @mmul_accum_34__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem36 = AIE.objectFifo.subview.access %subview36[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview37 = AIE.objectFifo.acquire @mmul_accum_33__pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem37 = AIE.objectFifo.subview.access %subview37[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_accum(%elem34, %elem31, %elem36, %elem37) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_21(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_31___mmul_accum_33___pB(Consume, 1)
         AIE.objectFifo.release @mmul_accum_34__pC(Consume, 1)
         AIE.objectFifo.release @mmul_accum_33__pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_accum.o" }

   AIE.core(%tile32) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %arg3 = %c0 to %intmax step %c1 {
         %subview34 = AIE.objectFifo.acquire @mtbuffer_0__MTout_31(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem34 = AIE.objectFifo.subview.access %subview34[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview30 = AIE.objectFifo.acquire @mtbuffer_1___MTout_31___mmul_end_32___pB(Consume, 1) : !AIE.objectFifoSubview<memref<1024xi32>>
         %elem30 = AIE.objectFifo.subview.access %subview30[0] : !AIE.objectFifoSubview<memref<1024xi32>> -> memref<1024xi32>
         %subview37 = AIE.objectFifo.acquire @mmul_accum_33__pC(Consume, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem37 = AIE.objectFifo.subview.access %subview37[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>
         %subview38 = AIE.objectFifo.acquire @mmul_end_32___pC(Produce, 1) : !AIE.objectFifoSubview<memref<4096xi32>>
         %elem38 = AIE.objectFifo.subview.access %subview38[0] : !AIE.objectFifoSubview<memref<4096xi32>> -> memref<4096xi32>

         func.call @mmul_end(%elem34, %elem30, %elem37, %elem38) : (memref<1024xi32>, memref<1024xi32>, memref<4096xi32>, memref<4096xi32>) -> ()

         AIE.objectFifo.release @mtbuffer_0__MTout_31(Consume, 1)
         AIE.objectFifo.release @mtbuffer_1___MTout_31___mmul_end_32___pB(Consume, 1)
         AIE.objectFifo.release @mmul_accum_33__pC(Consume, 1)
         AIE.objectFifo.release @mmul_end_32___pC(Produce, 1)
      }
      AIE.end
   } { link_with="mmul_end.o" }

func.func @sequence(%itbuffer_0 : memref<64x64xi32>,%itbuffer_1 : memref<256x64xi32>,%itbuffer_2 : memref<64x256xi32>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32


    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_2[%c0, %c0, %c0, %c0][%c1, %c1, %c64, %c64][%c0, %c0, %c256]){ metadata= @mtbuffer_2___MTout___itbuffer_2___ITin_00, id = 2 : i32 } :(i32, i32, memref<64x256xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c0][%c1, %c1, %c64, %c16][%c0, %c0, %c64]){ metadata= @itbuffer_0___ITout_00___mtbuffer_0___MTin_01, id = 1 : i32 } :(i32, i32, memref<64x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c0][%c1, %c1, %c256, %c16][%c0, %c0, %c64]){ metadata= @itbuffer_1___ITout_00___mtbuffer_1___MTin_01, id = 0 : i32 } :(i32, i32, memref<256x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_2[%c0, %c0, %c0, %c64][%c1, %c1, %c64, %c64][%c0, %c0, %c256]){ metadata= @mtbuffer_2___MTout___itbuffer_2___ITin_10, id = 5 : i32 } :(i32, i32, memref<64x256xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c16][%c1, %c1, %c64, %c16][%c0, %c0, %c64]){ metadata= @itbuffer_0___ITout_10___mtbuffer_0___MTin_11, id = 4 : i32 } :(i32, i32, memref<64x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c16][%c1, %c1, %c256, %c16][%c0, %c0, %c64]){ metadata= @itbuffer_1___ITout_10___mtbuffer_1___MTin_11, id = 3 : i32 } :(i32, i32, memref<256x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
    
    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_2[%c0, %c0, %c0, %c128][%c1, %c1, %c64, %c64][%c0, %c0, %c256]){ metadata= @mtbuffer_2___MTout___itbuffer_2___ITin_20, id = 8 : i32 } :(i32, i32, memref<64x256xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c32][%c1, %c1, %c64, %c16][%c0, %c0, %c64]){ metadata= @itbuffer_0___ITout_20___mtbuffer_0___MTin_21, id = 7 : i32 } :(i32, i32, memref<64x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c32][%c1, %c1, %c256, %c16][%c0, %c0, %c64]){ metadata= @itbuffer_1___ITout_20___mtbuffer_1___MTin_21, id = 6 : i32 } :(i32, i32, memref<256x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])
    
    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_2[%c0, %c0, %c0, %c192][%c1, %c1, %c64, %c64][%c0, %c0, %c256]){ metadata= @mtbuffer_2___MTout___itbuffer_2___ITin_30, id = 11 : i32 } :(i32, i32, memref<64x256xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_0[%c0, %c0, %c0, %c48][%c1, %c1, %c64, %c16][%c0, %c0, %c64]){ metadata= @itbuffer_0___ITout_30___mtbuffer_0___MTin_31, id = 10 : i32 } :(i32, i32, memref<64x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.dma_memcpy_nd(%c0, %c0,%itbuffer_1[%c0, %c0, %c0, %c48][%c1, %c1, %c256, %c16][%c0, %c0, %c64]){ metadata= @itbuffer_1___ITout_30___mtbuffer_1___MTin_31, id = 9 : i32 } :(i32, i32, memref<256x64xi32>, [i32,i32,i32,i32], [i32,i32,i32,i32], [i32,i32,i32])

    AIEX.ipu.sync {column = 0 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    AIEX.ipu.sync {column = 1 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    AIEX.ipu.sync {column = 2 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    AIEX.ipu.sync {column = 3 : i32, row = 0 : i32, direction = 0 : i32, channel = 0 : i32, column_num = 1 : i32, row_num = 1 : i32 }
    return
}
 }
}
