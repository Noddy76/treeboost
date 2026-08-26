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

func main() {
	generateWeightedSumAVX2()
	generateWeightedSumContiguousAVX2()
	Generate()
}
