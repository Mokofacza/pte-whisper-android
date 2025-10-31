package com.example.ptewhisper

import android.util.Log
import kotlinx.coroutines.withTimeout
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

object ExecuTorchSafe {
    private const val TAG = "ExecuTorchSafe"

    // Prosta walidacja mel-spectrogramu Whisper: [1, 80, N]
    fun validateMelShape(shape: LongArray): Boolean {
        return shape.size == 3 && shape[0] == 1L && shape[1] == 80L && shape[2] > 0
    }

    sealed class InferenceResult {
        data class Success(val outputs: Array<EValue>) : InferenceResult()
        data class Error(val code: Code, val message: String, val cause: Throwable? = null) : InferenceResult()
    }

    enum class Code { NOT_LOADED, BAD_INPUT, EMPTY_OUTPUT, RUNTIME }

    /**
     * Wersja z timeoutem — wywołuj w korutynie.
     */
    suspend fun forwardWithTimeout(
        module: Module?,
        inputs: List<EValue>,
        method: String? = "forward",
        timeoutMs: Long = 5_000L
    ): InferenceResult {
        if (module == null) return InferenceResult.Error(Code.NOT_LOADED, "Module == null")
        if (inputs.isEmpty()) return InferenceResult.Error(Code.BAD_INPUT, "No inputs")

        return try {
            withTimeout(timeoutMs) {
                // (opcjonalnie) pre-load konkretnej metody
                if (method != null) runCatching { module.loadMethod(method) }

                val outs = module.forward(*inputs.toTypedArray())
                if (outs.isEmpty()) InferenceResult.Error(Code.EMPTY_OUTPUT, "No outputs")
                else InferenceResult.Success(outs)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "forward error: ${t.message}", t)
            InferenceResult.Error(Code.RUNTIME, t.message ?: "Unknown", t)
        }
    }

    /**
     * Wersja blokująca (bez timeoutu) — prosta obsługa try/catch.
     */
    fun forwardBlocking(
        module: Module?,
        inputs: List<EValue>,
        method: String? = "forward"
    ): InferenceResult {
        if (module == null) return InferenceResult.Error(Code.NOT_LOADED, "Module == null")
        if (inputs.isEmpty()) return InferenceResult.Error(Code.BAD_INPUT, "No inputs")

        return try {
            if (method != null) runCatching { module.loadMethod(method) }
            val outs = module.forward(*inputs.toTypedArray())
            if (outs.isEmpty()) InferenceResult.Error(Code.EMPTY_OUTPUT, "No outputs")
            else InferenceResult.Success(outs)
        } catch (t: Throwable) {
            Log.e(TAG, "forward error: ${t.message}", t)
            InferenceResult.Error(Code.RUNTIME, t.message ?: "Unknown", t)
        }
    }

    /**
     * Bezpieczny odczyt pierwszego tensora z wyjścia.
     */
    fun firstTensorOrNull(outputs: Array<EValue>): Tensor? = try {
        outputs.firstOrNull()?.toTensor()
    } catch (_: Throwable) { null }
}
