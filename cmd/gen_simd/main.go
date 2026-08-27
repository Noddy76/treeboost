package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
)

func generateWeightedSumAVX2() {
	TEXT("computeWeightedSumAVX2", NOSPLIT, "func(W []float64, Y []float64, indices []int) (sumW, sumWY float64)")
	Doc("computeWeightedSumAVX2 computes sumW and sumWY using AVX2 FMA 256-bit SIMD instructions with indirect indices.")

	wPtr := Load(Param("W").Base(), GP64())
	yPtr := Load(Param("Y").Base(), GP64())
	idxPtr := Load(Param("indices").Base(), GP64())
	n := Load(Param("indices").Len(), GP64())

	sumWAcc := YMM()
	sumWYAcc := YMM()
	VZEROUPPER()
	VXORPD(sumWAcc, sumWAcc, sumWAcc)
	VXORPD(sumWYAcc, sumWYAcc, sumWYAcc)

	idx := GP64()
	XORQ(idx, idx)

	remInit := GP64()
	MOVQ(n, remInit)
	CMPQ(remInit, U8(4))
	JLT(LabelRef("tail"))

	Label("loop")

	i0 := GP64()
	i1 := GP64()
	i2 := GP64()
	i3 := GP64()

	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8}, i0)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 8}, i1)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 16}, i2)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 24}, i3)

	wVec := YMM()
	w1 := XMM()
	wHigh := XMM()
	w3 := XMM()

	VMOVSD(Mem{Base: wPtr, Index: i0, Scale: 8}, wVec.AsX())
	VMOVSD(Mem{Base: wPtr, Index: i1, Scale: 8}, w1)
	VUNPCKLPD(w1, wVec.AsX(), wVec.AsX())

	VMOVSD(Mem{Base: wPtr, Index: i2, Scale: 8}, wHigh)
	VMOVSD(Mem{Base: wPtr, Index: i3, Scale: 8}, w3)
	VUNPCKLPD(w3, wHigh, wHigh)
	VINSERTF128(U8(1), wHigh, wVec, wVec)

	yVec := YMM()
	y1 := XMM()
	yHigh := XMM()
	y3 := XMM()

	VMOVSD(Mem{Base: yPtr, Index: i0, Scale: 8}, yVec.AsX())
	VMOVSD(Mem{Base: yPtr, Index: i1, Scale: 8}, y1)
	VUNPCKLPD(y1, yVec.AsX(), yVec.AsX())

	VMOVSD(Mem{Base: yPtr, Index: i2, Scale: 8}, yHigh)
	VMOVSD(Mem{Base: yPtr, Index: i3, Scale: 8}, y3)
	VUNPCKLPD(y3, yHigh, yHigh)
	VINSERTF128(U8(1), yHigh, yVec, yVec)

	VADDPD(wVec, sumWAcc, sumWAcc)
	VFMADD231PD(yVec, wVec, sumWYAcc)

	ADDQ(U8(4), idx)
	rem := GP64()
	MOVQ(n, rem)
	SUBQ(idx, rem)
	CMPQ(rem, U8(4))
	JGE(LabelRef("loop"))

	Label("tail")
	CMPQ(idx, n)
	JGE(LabelRef("reduce"))

	tailIdx := GP64()
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8}, tailIdx)

	wTailX := XMM()
	yTailX := XMM()
	VMOVSD(Mem{Base: wPtr, Index: tailIdx, Scale: 8}, wTailX)
	VMOVSD(Mem{Base: yPtr, Index: tailIdx, Scale: 8}, yTailX)

	wTailY := YMM()
	yTailY := YMM()
	VXORPD(wTailY, wTailY, wTailY)
	VXORPD(yTailY, yTailY, yTailY)
	VINSERTF128(U8(0), wTailX, wTailY, wTailY)
	VINSERTF128(U8(0), yTailX, yTailY, yTailY)

	VADDPD(wTailY, sumWAcc, sumWAcc)
	VFMADD231PD(yTailY, wTailY, sumWYAcc)

	INCQ(idx)
	JMP(LabelRef("tail"))

	Label("reduce")
	wHi := XMM()
	yHi := XMM()
	VEXTRACTF128(U8(1), sumWAcc, wHi)
	VEXTRACTF128(U8(1), sumWYAcc, yHi)

	wLo := sumWAcc.AsX()
	yLo := sumWYAcc.AsX()
	VADDPD(wHi, wLo, wLo)
	VADDPD(yHi, yLo, yLo)

	wUpper := XMM()
	yUpper := XMM()
	VUNPCKHPD(wLo, wLo, wUpper)
	VUNPCKHPD(yLo, yLo, yUpper)

	VADDSD(wUpper, wLo, wLo)
	VADDSD(yUpper, yLo, yLo)

	Store(wLo, Return("sumW"))
	Store(yLo, Return("sumWY"))

	VZEROUPPER()
	RET()
}

func generateWeightedSumContiguousAVX2() {
	TEXT("computeWeightedSumContiguousAVX2", NOSPLIT, "func(W []float64, Y []float64) (sumW, sumWY float64)")
	Doc("computeWeightedSumContiguousAVX2 computes sumW and sumWY using AVX2 FMA 256-bit SIMD instructions over contiguous memory.")

	wPtr := Load(Param("W").Base(), GP64())
	yPtr := Load(Param("Y").Base(), GP64())
	wLen := Load(Param("W").Len(), GP64())
	yLen := Load(Param("Y").Len(), GP64())

	n := GP64()
	MOVQ(wLen, n)
	CMPQ(yLen, n)
	CMOVQLT(yLen, n)

	sumW0 := YMM()
	sumW1 := YMM()
	sumWY0 := YMM()
	sumWY1 := YMM()

	VZEROUPPER()
	VXORPD(sumW0, sumW0, sumW0)
	VXORPD(sumW1, sumW1, sumW1)
	VXORPD(sumWY0, sumWY0, sumWY0)
	VXORPD(sumWY1, sumWY1, sumWY1)

	idx := GP64()
	XORQ(idx, idx)

	rem8 := GP64()
	MOVQ(n, rem8)
	CMPQ(rem8, U8(8))
	JLT(LabelRef("loop4_check"))

	Label("loop8")
	wVec0 := YMM()
	yVec0 := YMM()
	wVec1 := YMM()
	yVec1 := YMM()

	VMOVUPD(Mem{Base: wPtr, Index: idx, Scale: 8, Disp: 0}, wVec0)
	VMOVUPD(Mem{Base: yPtr, Index: idx, Scale: 8, Disp: 0}, yVec0)
	VADDPD(wVec0, sumW0, sumW0)
	VFMADD231PD(yVec0, wVec0, sumWY0)

	VMOVUPD(Mem{Base: wPtr, Index: idx, Scale: 8, Disp: 32}, wVec1)
	VMOVUPD(Mem{Base: yPtr, Index: idx, Scale: 8, Disp: 32}, yVec1)
	VADDPD(wVec1, sumW1, sumW1)
	VFMADD231PD(yVec1, wVec1, sumWY1)

	ADDQ(U8(8), idx)
	rem8Check := GP64()
	MOVQ(n, rem8Check)
	SUBQ(idx, rem8Check)
	CMPQ(rem8Check, U8(8))
	JGE(LabelRef("loop8"))

	Label("loop4_check")
	rem4Check := GP64()
	MOVQ(n, rem4Check)
	SUBQ(idx, rem4Check)
	CMPQ(rem4Check, U8(4))
	JLT(LabelRef("tail_check"))

	Label("loop4")
	wVec4 := YMM()
	yVec4 := YMM()
	VMOVUPD(Mem{Base: wPtr, Index: idx, Scale: 8}, wVec4)
	VMOVUPD(Mem{Base: yPtr, Index: idx, Scale: 8}, yVec4)
	VADDPD(wVec4, sumW0, sumW0)
	VFMADD231PD(yVec4, wVec4, sumWY0)
	ADDQ(U8(4), idx)

	Label("tail_check")
	CMPQ(idx, n)
	JGE(LabelRef("reduce"))

	Label("tail_loop")
	wTailX := XMM()
	yTailX := XMM()
	VMOVSD(Mem{Base: wPtr, Index: idx, Scale: 8}, wTailX)
	VMOVSD(Mem{Base: yPtr, Index: idx, Scale: 8}, yTailX)
	wTailY := YMM()
	yTailY := YMM()
	VXORPD(wTailY, wTailY, wTailY)
	VXORPD(yTailY, yTailY, yTailY)
	VINSERTF128(U8(0), wTailX, wTailY, wTailY)
	VINSERTF128(U8(0), yTailX, yTailY, yTailY)
	VADDPD(wTailY, sumW0, sumW0)
	VFMADD231PD(yTailY, wTailY, sumWY0)
	INCQ(idx)
	CMPQ(idx, n)
	JLT(LabelRef("tail_loop"))

	Label("reduce")
	VADDPD(sumW1, sumW0, sumW0)
	VADDPD(sumWY1, sumWY0, sumWY0)

	wHi := XMM()
	yHi := XMM()
	VEXTRACTF128(U8(1), sumW0, wHi)
	VEXTRACTF128(U8(1), sumWY0, yHi)

	wLo := sumW0.AsX()
	yLo := sumWY0.AsX()
	VADDPD(wHi, wLo, wLo)
	VADDPD(yHi, yLo, yLo)

	wUpper := XMM()
	yUpper := XMM()
	VUNPCKHPD(wLo, wLo, wUpper)
	VUNPCKHPD(yLo, yLo, yUpper)

	VADDSD(wUpper, wLo, wLo)
	VADDSD(yUpper, yLo, yLo)

	Store(wLo, Return("sumW"))
	Store(yLo, Return("sumWY"))

	VZEROUPPER()
	RET()
}

func generateWeightedSumAVX512() {
	TEXT("computeWeightedSumAVX512", NOSPLIT, "func(W []float64, Y []float64, indices []int) (sumW, sumWY float64)")
	Doc("computeWeightedSumAVX512 computes sumW and sumWY using AVX-512 512-bit SIMD instructions with indirect indices.")

	wPtr := Load(Param("W").Base(), GP64())
	yPtr := Load(Param("Y").Base(), GP64())
	idxPtr := Load(Param("indices").Base(), GP64())
	n := Load(Param("indices").Len(), GP64())

	sumWAcc := ZMM()
	sumWYAcc := ZMM()
	VZEROUPPER()
	VPXORD(sumWAcc, sumWAcc, sumWAcc)
	VPXORD(sumWYAcc, sumWYAcc, sumWYAcc)

	idx := GP64()
	XORQ(idx, idx)

	remInit := GP64()
	MOVQ(n, remInit)
	CMPQ(remInit, U8(8))
	JLT(LabelRef("tail"))

	Label("loop")

	i0 := GP64()
	i1 := GP64()
	i2 := GP64()
	i3 := GP64()
	i4 := GP64()
	i5 := GP64()
	i6 := GP64()
	i7 := GP64()

	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 0}, i0)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 8}, i1)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 16}, i2)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 24}, i3)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 32}, i4)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 40}, i5)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 48}, i6)
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8, Disp: 56}, i7)

	wLoY := YMM()
	wHiY := YMM()
	w1 := XMM()
	w2 := XMM()
	w3 := XMM()
	w5 := XMM()
	w6 := XMM()
	w7 := XMM()

	VMOVSD(Mem{Base: wPtr, Index: i0, Scale: 8}, wLoY.AsX())
	VMOVSD(Mem{Base: wPtr, Index: i1, Scale: 8}, w1)
	VUNPCKLPD(w1, wLoY.AsX(), wLoY.AsX())
	VMOVSD(Mem{Base: wPtr, Index: i2, Scale: 8}, w2)
	VMOVSD(Mem{Base: wPtr, Index: i3, Scale: 8}, w3)
	VUNPCKLPD(w3, w2, w2)
	VINSERTF128(U8(1), w2, wLoY, wLoY)

	VMOVSD(Mem{Base: wPtr, Index: i4, Scale: 8}, wHiY.AsX())
	VMOVSD(Mem{Base: wPtr, Index: i5, Scale: 8}, w5)
	VUNPCKLPD(w5, wHiY.AsX(), wHiY.AsX())
	VMOVSD(Mem{Base: wPtr, Index: i6, Scale: 8}, w6)
	VMOVSD(Mem{Base: wPtr, Index: i7, Scale: 8}, w7)
	VUNPCKLPD(w7, w6, w6)
	VINSERTF128(U8(1), w6, wHiY, wHiY)

	wVec := ZMM()
	VINSERTF64X4(U8(0), wLoY, wVec, wVec)
	VINSERTF64X4(U8(1), wHiY, wVec, wVec)

	yLoY := YMM()
	yHiY := YMM()
	y1 := XMM()
	y2 := XMM()
	y3 := XMM()
	y5 := XMM()
	y6 := XMM()
	y7 := XMM()

	VMOVSD(Mem{Base: yPtr, Index: i0, Scale: 8}, yLoY.AsX())
	VMOVSD(Mem{Base: yPtr, Index: i1, Scale: 8}, y1)
	VUNPCKLPD(y1, yLoY.AsX(), yLoY.AsX())
	VMOVSD(Mem{Base: yPtr, Index: i2, Scale: 8}, y2)
	VMOVSD(Mem{Base: yPtr, Index: i3, Scale: 8}, y3)
	VUNPCKLPD(y3, y2, y2)
	VINSERTF128(U8(1), y2, yLoY, yLoY)

	VMOVSD(Mem{Base: yPtr, Index: i4, Scale: 8}, yHiY.AsX())
	VMOVSD(Mem{Base: yPtr, Index: i5, Scale: 8}, y5)
	VUNPCKLPD(y5, yHiY.AsX(), yHiY.AsX())
	VMOVSD(Mem{Base: yPtr, Index: i6, Scale: 8}, y6)
	VMOVSD(Mem{Base: yPtr, Index: i7, Scale: 8}, y7)
	VUNPCKLPD(y7, y6, y6)
	VINSERTF128(U8(1), y6, yHiY, yHiY)

	yVec := ZMM()
	VINSERTF64X4(U8(0), yLoY, yVec, yVec)
	VINSERTF64X4(U8(1), yHiY, yVec, yVec)

	VADDPD(wVec, sumWAcc, sumWAcc)
	VFMADD231PD(yVec, wVec, sumWYAcc)

	ADDQ(U8(8), idx)
	rem := GP64()
	MOVQ(n, rem)
	SUBQ(idx, rem)
	CMPQ(rem, U8(8))
	JGE(LabelRef("loop"))

	Label("tail")
	CMPQ(idx, n)
	JGE(LabelRef("reduce"))

	tailIdx := GP64()
	MOVQ(Mem{Base: idxPtr, Index: idx, Scale: 8}, tailIdx)

	wTailX := XMM()
	yTailX := XMM()
	VMOVSD(Mem{Base: wPtr, Index: tailIdx, Scale: 8}, wTailX)
	VMOVSD(Mem{Base: yPtr, Index: tailIdx, Scale: 8}, yTailX)

	VADDSD(wTailX, sumWAcc.AsX(), sumWAcc.AsX())
	VFMADD231SD(yTailX, wTailX, sumWYAcc.AsX())

	INCQ(idx)
	JMP(LabelRef("tail"))

	Label("reduce")
	wRedHiY := YMM()
	yRedHiY := YMM()
	VEXTRACTF64X4(U8(1), sumWAcc, wRedHiY)
	VEXTRACTF64X4(U8(1), sumWYAcc, yRedHiY)

	VADDPD(wRedHiY, sumWAcc.AsY(), sumWAcc.AsY())
	VADDPD(yRedHiY, sumWYAcc.AsY(), sumWYAcc.AsY())

	wHiX := XMM()
	yHiX := XMM()
	VEXTRACTF128(U8(1), sumWAcc.AsY(), wHiX)
	VEXTRACTF128(U8(1), sumWYAcc.AsY(), yHiX)

	VADDPD(wHiX, sumWAcc.AsX(), sumWAcc.AsX())
	VADDPD(yHiX, sumWYAcc.AsX(), sumWYAcc.AsX())

	wUpper := XMM()
	yUpper := XMM()
	VUNPCKHPD(sumWAcc.AsX(), sumWAcc.AsX(), wUpper)
	VUNPCKHPD(sumWYAcc.AsX(), sumWYAcc.AsX(), yUpper)

	VADDSD(wUpper, sumWAcc.AsX(), sumWAcc.AsX())
	VADDSD(yUpper, sumWYAcc.AsX(), sumWYAcc.AsX())

	Store(sumWAcc.AsX(), Return("sumW"))
	Store(sumWYAcc.AsX(), Return("sumWY"))

	VZEROUPPER()
	RET()
}

func generateWeightedSumContiguousAVX512() {
	TEXT("computeWeightedSumContiguousAVX512", NOSPLIT, "func(W []float64, Y []float64) (sumW, sumWY float64)")
	Doc("computeWeightedSumContiguousAVX512 computes sumW and sumWY using AVX-512 512-bit SIMD instructions over contiguous memory.")

	wPtr := Load(Param("W").Base(), GP64())
	yPtr := Load(Param("Y").Base(), GP64())
	wLen := Load(Param("W").Len(), GP64())
	yLen := Load(Param("Y").Len(), GP64())

	n := GP64()
	MOVQ(wLen, n)
	CMPQ(yLen, n)
	CMOVQLT(yLen, n)

	sumW0 := ZMM()
	sumW1 := ZMM()
	sumWY0 := ZMM()
	sumWY1 := ZMM()

	VZEROUPPER()
	VPXORD(sumW0, sumW0, sumW0)
	VPXORD(sumW1, sumW1, sumW1)
	VPXORD(sumWY0, sumWY0, sumWY0)
	VPXORD(sumWY1, sumWY1, sumWY1)

	idx := GP64()
	XORQ(idx, idx)

	rem16 := GP64()
	MOVQ(n, rem16)
	CMPQ(rem16, U8(16))
	JLT(LabelRef("loop8_check"))

	Label("loop16")
	wVec0 := ZMM()
	yVec0 := ZMM()
	wVec1 := ZMM()
	yVec1 := ZMM()

	VMOVUPD(Mem{Base: wPtr, Index: idx, Scale: 8, Disp: 0}, wVec0)
	VMOVUPD(Mem{Base: yPtr, Index: idx, Scale: 8, Disp: 0}, yVec0)
	VADDPD(wVec0, sumW0, sumW0)
	VFMADD231PD(yVec0, wVec0, sumWY0)

	VMOVUPD(Mem{Base: wPtr, Index: idx, Scale: 8, Disp: 64}, wVec1)
	VMOVUPD(Mem{Base: yPtr, Index: idx, Scale: 8, Disp: 64}, yVec1)
	VADDPD(wVec1, sumW1, sumW1)
	VFMADD231PD(yVec1, wVec1, sumWY1)

	ADDQ(U8(16), idx)
	rem16Check := GP64()
	MOVQ(n, rem16Check)
	SUBQ(idx, rem16Check)
	CMPQ(rem16Check, U8(16))
	JGE(LabelRef("loop16"))

	Label("loop8_check")
	rem8Check := GP64()
	MOVQ(n, rem8Check)
	SUBQ(idx, rem8Check)
	CMPQ(rem8Check, U8(8))
	JLT(LabelRef("tail_check"))

	Label("loop8")
	wVec8 := ZMM()
	yVec8 := ZMM()
	VMOVUPD(Mem{Base: wPtr, Index: idx, Scale: 8}, wVec8)
	VMOVUPD(Mem{Base: yPtr, Index: idx, Scale: 8}, yVec8)
	VADDPD(wVec8, sumW0, sumW0)
	VFMADD231PD(yVec8, wVec8, sumWY0)
	ADDQ(U8(8), idx)

	Label("tail_check")
	CMPQ(idx, n)
	JGE(LabelRef("reduce"))

	Label("tail_loop")
	wTailX := XMM()
	yTailX := XMM()
	VMOVSD(Mem{Base: wPtr, Index: idx, Scale: 8}, wTailX)
	VMOVSD(Mem{Base: yPtr, Index: idx, Scale: 8}, yTailX)
	VADDSD(wTailX, sumW0.AsX(), sumW0.AsX())
	VFMADD231SD(yTailX, wTailX, sumWY0.AsX())
	INCQ(idx)
	CMPQ(idx, n)
	JLT(LabelRef("tail_loop"))

	Label("reduce")
	VADDPD(sumW1, sumW0, sumW0)
	VADDPD(sumWY1, sumWY0, sumWY0)

	wHiY := YMM()
	yHiY := YMM()
	VEXTRACTF64X4(U8(1), sumW0, wHiY)
	VEXTRACTF64X4(U8(1), sumWY0, yHiY)

	VADDPD(wHiY, sumW0.AsY(), sumW0.AsY())
	VADDPD(yHiY, sumWY0.AsY(), sumWY0.AsY())

	wHiX := XMM()
	yHiX := XMM()
	VEXTRACTF128(U8(1), sumW0.AsY(), wHiX)
	VEXTRACTF128(U8(1), sumWY0.AsY(), yHiX)

	VADDPD(wHiX, sumW0.AsX(), sumW0.AsX())
	VADDPD(yHiX, sumWY0.AsX(), sumWY0.AsX())

	wUpper := XMM()
	yUpper := XMM()
	VUNPCKHPD(sumW0.AsX(), sumW0.AsX(), wUpper)
	VUNPCKHPD(sumWY0.AsX(), sumWY0.AsX(), yUpper)

	VADDSD(wUpper, sumW0.AsX(), sumW0.AsX())
	VADDSD(yUpper, sumWY0.AsX(), sumWY0.AsX())

	Store(sumW0.AsX(), Return("sumW"))
	Store(sumWY0.AsX(), Return("sumWY"))

	VZEROUPPER()
	RET()
}

func main() {
	generateWeightedSumAVX512()
	generateWeightedSumContiguousAVX512()
	generateWeightedSumAVX2()
	generateWeightedSumContiguousAVX2()
	Generate()
}
