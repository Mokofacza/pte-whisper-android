package com.example.ptewhisper

import android.Manifest
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.annotation.RequiresPermission
import kotlin.math.abs
import kotlin.math.max

class AudioMic(
    private val sampleRate: Int = 16_000
) {
    private var record: AudioRecord? = null
    private var isRecording = false
    private val pcm = ArrayList<Short>(sampleRate * 35) // bufor ~35 s

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun start() {
        if (isRecording) return
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        record = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            max(minBuf, sampleRate) // >= 1 sek
        )
        pcm.clear()
        record?.startRecording()
        isRecording = true

        Thread {
            val buf = ShortArray(2048)
            while (isRecording) {
                val n = record?.read(buf, 0, buf.size) ?: 0
                if (n > 0) {
                    for (i in 0 until n) pcm.add(buf[i])
                    // miękka ochrona: max 35s
                    if (pcm.size > sampleRate * 35) stop()
                }
            }
        }.start()
    }

    fun stop(): FloatArray {
        if (!isRecording) return FloatArray(0)
        isRecording = false
        record?.stop()
        record?.release()
        record = null

        // short -> float [-1,1]
        val out = FloatArray(pcm.size)
        for (i in pcm.indices) out[i] = pcm[i] / 32768f
        return out
    }

    fun isActive(): Boolean = isRecording

    /** Prosty wskaźnik głośności do UI */
    fun level(): Float {
        var m = 0
        val r = record ?: return 0f
        val tmp = ShortArray(128)
        val n = r.read(tmp, 0, tmp.size, AudioRecord.READ_NON_BLOCKING)
        if (n > 0) for (i in 0 until n) m = max(m, abs(tmp[i].toInt()))
        return m / 32768f
    }
}
