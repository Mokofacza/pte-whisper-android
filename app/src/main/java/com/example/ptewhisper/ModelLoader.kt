package com.example.ptewhisper

import android.content.Context
import android.util.Log
import org.pytorch.executorch.Module
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class ModelLoader(private val context: Context) {

    companion object {
        private const val TAG = "WhisperModelLoader"
    }

    /** Wypisz co faktycznie masz w assets (pomocne w debug). */
    fun listAssets(prefix: String = ""): List<String> {
        return try {
            context.assets.list(prefix)?.toList()?.sorted()?.map {
                if (prefix.isNotEmpty()) "$prefix/$it" else it
            } ?: emptyList()
        } catch (e: Exception) {
            Log.e(TAG, "assets.list('$prefix') error: ${e.message}", e)
            emptyList()
        }
    }

    /** Skopiuj asset do prywatnego storage i zwróć pełną ścieżkę. Tworzy potrzebne katalogi. */
    private fun materializeAsset(assetName: String, overwrite: Boolean = false): String? {
        val outFile = File(context.filesDir, assetName) // może mieć podkatalogi
        try {
            // Upewnij się, że katalogi istnieją
            outFile.parentFile?.let { parent ->
                if (!parent.exists() && !parent.mkdirs()) {
                    Log.e(TAG, "Nie mogę utworzyć katalogu: ${parent.absolutePath}")
                    return null
                }
            }

            if (overwrite || !outFile.exists() || outFile.length() == 0L) {
                context.assets.open(assetName).use { inStream ->
                    FileOutputStream(outFile, false).use { outStream ->
                        val buf = ByteArray(16 * 1024)
                        while (true) {
                            val n = inStream.read(buf)
                            if (n <= 0) break
                            outStream.write(buf, 0, n)
                        }
                        outStream.flush()
                    }
                }
                Log.d(TAG, "Skopiowano asset '$assetName' -> ${outFile.absolutePath} (${outFile.length()} B)")
            } else {
                Log.d(TAG, "Asset już jest: ${outFile.absolutePath} (${outFile.length()} B)")
            }
            return outFile.absolutePath
        } catch (e: IOException) {
            Log.e(TAG, "Kopiowanie assetu '$assetName' nieudane: ${e.message}", e)
            return null
        }
    }

    /** Załaduj ExecuTorch Module z .pte (najpierw MMAP, w razie czego fallback do FILE). */
    fun loadModuleFromAssets(assetName: String): Module? {
        val modelPath = materializeAsset(assetName) ?: return null
        return try {
            // Szybka ścieżka: MMAP
            val threads = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
            try {
                Module.load(modelPath, Module.LOAD_MODE_MMAP, threads).also {
                    it.loadMethod("forward")
                    Log.i(TAG, "Załadowano (MMAP): $assetName")
                }
            } catch (mmapErr: Throwable) {
                Log.w(TAG, "MMAP nie działa dla $assetName: ${mmapErr.message}. Próba LOAD_MODE_FILE…")
                Module.load(modelPath, Module.LOAD_MODE_FILE, threads).also {
                    it.loadMethod("forward")
                    Log.i(TAG, "Załadowano (FILE): $assetName")
                }
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Nie udało się załadować $assetName: ${e.message}", e)
            null
        }
    }
}
