#include <aie_api/aie.hpp>
extern "C"
{
    void mmul_start(uint8_t *__restrict pA, uint8_t *__restrict pB, uint16_t *__restrict pC)
    {
        const int shift = 0;
        const int M = 8;
        const int K = 8;
        const int N = 4;
        const int rowA = 64 / M;
        const int colA = 64 / K;
        const int colB = 64 / N;
        using MMUL = ::aie::mmul<M, K, N, uint8_t, uint8_t, acc32>;

        for (unsigned z = 0; z < rowA; z += 1)
            chess_prepare_for_pipelining
            {
                uint16_t *__restrict pC1 = pC + (z * colB + 0) * MMUL::size_C;
                for (unsigned j = 0; j < colB; j += 2)
                    chess_prepare_for_pipelining
                    {
                        const uint8_t *__restrict pA1 = pA + (z * colA + 0) * MMUL::size_A;
                        const uint8_t *__restrict pB1 = pB + (0 * colB + j) * MMUL::size_B;
                        const uint8_t *__restrict pB2 = pB + (0 * colB + (j + 1)) * MMUL::size_B;
                        ::aie::vector<uint8_t, MMUL::size_A> A0 = ::aie::load_v<MMUL::size_A>(pA1);
                        pA1 += MMUL::size_A;
                        ::aie::vector<uint8_t, MMUL::size_B> B0 = ::aie::load_v<MMUL::size_B>(pB1);
                        pB1 += MMUL::size_B * colB;
                        ::aie::vector<uint8_t, MMUL::size_B> B1 = ::aie::load_v<MMUL::size_B>(pB2);
                        pB2 += MMUL::size_B * colB;

                        MMUL C00;
                        C00.mul(A0, B0);
                        MMUL C01;
                        C01.mul(A0, B1);

                        for (unsigned i = 1; i < colA; ++i)
                            chess_prepare_for_pipelining
                            {
                                A0 = ::aie::load_v<MMUL::size_A>(pA1);
                                pA1 += MMUL::size_A;
                                B0 = ::aie::load_v<MMUL::size_B>(pB1);
                                pB1 += MMUL::size_B * colB;
                                B1 = ::aie::load_v<MMUL::size_B>(pB2);
                                pB2 += MMUL::size_B * colB;

                                C00.mac(A0, B0);
                                C01.mac(A0, B1);
                            }

                        ::aie::store_v(pC1, C00.template to_vector<uint16_t>(shift));
                        pC1 += MMUL::size_C;
                        ::aie::store_v(pC1, C01.template to_vector<uint16_t>(shift));
                        pC1 += MMUL::size_C;
                    }
            }
    }
}