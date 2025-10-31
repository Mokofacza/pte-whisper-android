package com.example.ptewhisper

import kotlin.math.*

/** Prosty FFT radix-2 (N=512). Zwraca pary (re, im) in-place. */
private object FFT512 {
    private const val N = 512
    private val cosTable = DoubleArray(N / 2)
    private val sinTable = DoubleArray(N / 2)
    init {
        for (i in 0 until N / 2) {
            val ang = -2.0 * Math.PI * i / N
            cosTable[i] = cos(ang)
            sinTable[i] = sin(ang)
        }
    }
    fun fftReal(input: FloatArray, offset: Int, outRe: DoubleArray, outIm: DoubleArray) {
        // kopiuj i zeruj im
        for (i in 0 until N) { outRe[i] = input[offset + i].toDouble(); outIm[i] = 0.0 }
        // bit reverse
        var j = 0
        for (i in 1 until N - 1) {
            var bit = N shr 1
            while (j >= bit) { j -= bit; bit = bit shr 1 }
            j += bit
            if (i < j) {
                val tr = outRe[i]; outRe[i] = outRe[j]; outRe[j] = tr
                val ti = outIm[i]; outIm[i] = outIm[j]; outIm[j] = ti
            }
        }
        // iteracje
        var len = 2
        while (len <= N) {
            val step = N / len
            val half = len shr 1
            var i = 0
            while (i < N) {
                var k = 0
                for (j2 in 0 until half) {
                    val idx = k
                    val wr = cosTable[idx]; val wi = sinTable[idx]
                    val uRe = outRe[i + j2]; val uIm = outIm[i + j2]
                    val vRe = outRe[i + j2 + half]; val vIm = outIm[i + j2 + half]
                    val tRe = wr * vRe - wi * vIm
                    val tIm = wr * vIm + wi * vRe
                    outRe[i + j2] = uRe + tRe
                    outIm[i + j2] = uIm + tIm
                    outRe[i + j2 + half] = uRe - tRe
                    outIm[i + j2 + half] = uIm - tIm
                    k += step
                }
                i += len
            }
            len = len shl 1
        }
    }
}

class MelSpec(
    private val sampleRate: Int = 16_000,
    private val nFft: Int = 512,        // zero-pad z okna 400
    private val winLength: Int = 400,   // 25 ms
    private val hopLength: Int = 160,   // 10 ms
    private val nMels: Int = 80,
    private val fMin: Double = 0.0,
    private val fMax: Double = 8000.0
) {
    private val hann = FloatArray(winLength) { i ->
        (0.5 * (1.0 - cos(2.0 * Math.PI * i / (winLength - 1)))).toFloat()
    }

    // macierz filtrów mel: [nMels, nFft/2+1]
    private val melFilter: Array<DoubleArray> = run {
        val nFreqs = nFft / 2 + 1
        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val mels = DoubleArray(nMels + 2) { i -> melMin + (melMax - melMin) * i / (nMels + 1) }
        val hz = DoubleArray(nMels + 2) { i -> melToHz(mels[i]) }
        val bins = IntArray(nMels + 2) { i -> floor((nFft + 1) * hz[i] / sampleRate).toInt().coerceIn(0, nFreqs - 1) }

        Array(nMels) { m ->
            val f = DoubleArray(nFreqs)
            val left = bins[m]; val center = bins[m + 1]; val right = bins[m + 2]
            for (k in left until center) {
                f[k] = (k - left).toDouble() / (center - left).coerceAtLeast(1)
            }
            for (k in center until right) {
                f[k] = (right - k).toDouble() / (right - center).coerceAtLeast(1)
            }
            f
        }
    }

    private fun hzToMel(hz: Double) = 2595.0 * ln(1.0 + hz / 700.0) / ln(10.0)
    private fun melToHz(m: Double) = 700.0 * (10.0.pow(m / 2595.0) - 1.0)

    /** PCM 16k float[-1,1] -> log-mel [80, T] z pad/trunc do T=3000 */
    fun logMel(pcm: FloatArray): FloatArray {
        // normalizacja i limit długości do ~30 s
        val targetSamples = 30 * sampleRate
        val x = if (pcm.size >= targetSamples) pcm.copyOfRange(0, targetSamples) else {
            FloatArray(targetSamples).apply { System.arraycopy(pcm, 0, this, 0, pcm.size) }
        }

        val nFrames = 1 + (x.size - winLength) / hopLength
        val spec = Array(nFrames) { DoubleArray(nFft / 2 + 1) }

        val winPad = FloatArray(nFft) // 512
        val re = DoubleArray(nFft); val im = DoubleArray(nFft)

        var frame = 0
        var offset = 0
        while (offset + winLength <= x.size && frame < nFrames) {
            // window + zero pad
            for (i in 0 until winLength) winPad[i] = x[offset + i] * hann[i]
            for (i in winLength until nFft) winPad[i] = 0f

            FFT512.fftReal(winPad, 0, re, im)

            // magnitude^2 (tylko 0..nFft/2)
            val row = spec[frame]
            for (k in 0..(nFft / 2)) {
                val rr = re[k]; val ii = im[k]
                row[k] = rr * rr + ii * ii
            }
            frame++
            offset += hopLength
        }

        // mel proj + log
        val melT = 3000
        val out = FloatArray(nMels * melT) { (-11f) } // wypelnij bardzo małą wartością log
        val logEps = 1e-10

        val T = min(melT, spec.size)
        for (t in 0 until T) {
            val p = spec[t]
            for (m in 0 until nMels) {
                var e = 0.0
                val f = melFilter[m]
                for (k in f.indices) if (f[k] != 0.0) e += f[k] * p[k]
                val v = ln(e + logEps).toFloat()
                out[m * melT + t] = v
            }
        }
        return out // shape [80, 3000] spłaszczone (kolumny t idą co 1)
    }
}
