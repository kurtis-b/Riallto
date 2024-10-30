#include <aie_api/aie.hpp>
extern "C"
{
    void mmul_end(uint8_t *__restrict pA, uint8_t *__restrict pB, uint16_t *pAccum, uint16_t *__restrict pC)
    {
        const int M = 8;
        const int K = 8;
        const int N = 4;
        const int rowA = 64 / M;
        const int colA = 64 / K;
        const int colB = 64 / N;
        using MMUL = ::aie::mmul<M, K, N, uint8_t, uint8_t>;

        for (unsigned z = 0; z < rowA; z += 2)
            chess_prepare_for_pipelining
            {
                uint16_t *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
                uint16_t *__restrict pC2 = pC + ((z + 1) * colB + 0) * MMUL::size_C;
                uint16_t *__restrict pAccum1 = pAccum + (z * colB + 0) * MMUL::size_C;
                uint16_t *__restrict pAccum2 = pAccum + ((z + 1) * colB + 0) * MMUL::size_C;
                for (unsigned j = 0; j < colB; j += 2)
                    chess_prepare_for_pipelining
                    {
                        const uint8_t *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
                        const uint8_t *__restrict pA2 = pA + ((z + 1) * colA + 0) * MMUL::size_A;
                        const uint8_t *__restrict pB1 = pB + (0 * colB + j) * MMUL::size_B;
                        const uint8_t *__restrict pB2 = pB + (0 * colB + (j + 1)) * MMUL::size_B;
                        ::aie::vector<uint8_t, MMUL::size_A> A0 = ::aie::load_v<MMUL::size_A>(pA1);
                        pA1 += MMUL::size_A;
                        ::aie::vector<uint8_t, MMUL::size_A> A1 = ::aie::load_v<MMUL::size_A>(pA2);
                        pA2 += MMUL::size_A;
                        ::aie::vector<uint8_t, MMUL::size_B> B0 = ::aie::load_v<MMUL::size_B>(pB1);
                        pB1 += MMUL::size_B * colB;
                        ::aie::vector<uint8_t, MMUL::size_B> B1 = ::aie::load_v<MMUL::size_B>(pB2);
                        pB2 += MMUL::size_B * colB;

                        MMUL C00;
                        C00.mul(A0, B0);
                        MMUL C01;
                        C01.mul(A0, B1);
                        MMUL C10;
                        C10.mul(A1, B0);
                        MMUL C11;
                        C11.mul(A1, B1);

                        for (unsigned i = 1; i < colA; ++i)
                            chess_prepare_for_pipelining
                            {
                                A0 = ::aie::load_v<MMUL::size_A>(pA1);
                                pA1 += MMUL::size_A;
                                A1 = ::aie::load_v<MMUL::size_A>(pA2);
                                pA2 += MMUL::size_A;
                                B0 = ::aie::load_v<MMUL::size_B>(pB1);
                                pB1 += MMUL::size_B * colB;
                                B1 = ::aie::load_v<MMUL::size_B>(pB2);
                                pB2 += MMUL::size_B * colB;

                                C00.mac(A0, B0);
                                C01.mac(A0, B1);
                                C10.mac(A1, B0);
                                C11.mac(A1, B1);
                            }

                        ::aie::store_v(pC1, ::aie::add(::aie::load_v<MMUL::size_C>(pAccum1), C00.template to_vector<uint16_t>()));
                        pC1 += MMUL::size_C;
                        ::aie::store_v(pC1, ::aie::add(::aie::load_v<MMUL::size_C>(pAccum1), C01.template to_vector<uint16_t>()));
                        pC1 += MMUL::size_C;
                        ::aie::store_v(pC2, ::aie::add(::aie::load_v<MMUL::size_C>(pAccum2), C10.template to_vector<uint16_t>()));
                        pC2 += MMUL::size_C;
                        ::aie::store_v(pC2, ::aie::add(::aie::load_v<MMUL::size_C>(pAccum2), C11.template to_vector<uint16_t>()));
                        pC2 += MMUL::size_C;
                    }
            }
    }
}