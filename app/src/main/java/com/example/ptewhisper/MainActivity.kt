package com.example.ptewhisper

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresPermission
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.example.ptewhisper.ui.theme.PTEWhisperTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor

class MainActivity : ComponentActivity() {

    private val ENCODER_ASSET = "xnnpack/whisper_small_encoder_xnnpack.pte"
    private val DECODER_ASSET = "xnnpack/whisper_small_decoder_xnnpack.pte"

    private lateinit var loader: ModelLoader
    private var whisperEncoder: Module? = null
    private var whisperDecoder: Module? = null

    private lateinit var mic: AudioMic
    private lateinit var mel: MelSpec
    private lateinit var tokenizer: SimpleTokenizer

    private var ui by mutableStateOf(UiState())

    private val askMic = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        ui = if (granted) ui.copy(status = "Uprawnienie do mikrofonu OK")
        else ui.copy(status = "❌ Brak uprawnienia do mikrofonu")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        loader = ModelLoader(applicationContext)
        mic = AudioMic(16_000)
        mel = MelSpec(16_000)
        tokenizer = SimpleTokenizer(applicationContext)

        setContent {
            PTEWhisperTheme {
                Scaffold(
                    modifier = Modifier.fillMaxSize(),
                    contentWindowInsets = WindowInsets.safeDrawing
                ) { inner ->
                    App(
                        state = ui,
                        onPerms = { ensureMicPermission() },
                        onLoad = { loadModels() },
                        onStartRec = {
                            if (ActivityCompat.checkSelfPermission(
                                    this, Manifest.permission.RECORD_AUDIO
                                ) != PackageManager.PERMISSION_GRANTED
                            ) return@App
                            startMic()
                        },
                        onStopTranscribe = { stopAndTranscribe() },
                        onUnload = { unloadModels() },
                        modifier = Modifier
                            .padding(inner)
                            .fillMaxSize()
                            .padding(20.dp)
                    )
                }
            }
        }
    }

    // ---------------- Actions ----------------

    private fun ensureMicPermission() {
        val has = ContextCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED
        if (!has) askMic.launch(Manifest.permission.RECORD_AUDIO)
        else ui = ui.copy(status = "Mikrofon OK")
    }

    private fun loadModels() {
        if (ui.busy) return
        lifecycleScope.launch(Dispatchers.Main) {
            ui = ui.copy(busy = true, status = "Ładowanie modeli…", lastError = null)
            val (enc, dec) = withContext(Dispatchers.Default) {
                val e = loader.loadModuleFromAssets(ENCODER_ASSET)
                val d = loader.loadModuleFromAssets(DECODER_ASSET)
                e to d
            }
            if (enc == null || dec == null) {
                ui = ui.copy(busy = false, loaded = false, status = "❌ Nie udało się załadować modeli")
            } else {
                whisperEncoder = enc
                whisperDecoder = dec
                ui = ui.copy(busy = false, loaded = true, status = "✔ Modele gotowe")
            }
        }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    private fun startMic() {
        if (!ui.loaded) { ui = ui.copy(status = "Najpierw załaduj modele"); return }
        ensureMicPermission()
        mic.start()
        ui = ui.copy(recording = true, status = "🎙️ Nagrywam…")
    }

    private fun stopAndTranscribe() {
        if (!ui.recording) return
        ui = ui.copy(recording = false)
        val pcm = mic.stop()
        lifecycleScope.launch(Dispatchers.Main) {
            ui = ui.copy(busy = true, status = "Przetwarzam audio…")
            val text = withContext(Dispatchers.Default) { transcribeOnce(pcm) }
            ui = if (text != null) {
                ui.copy(busy = false, status = "✅ Transkrypcja gotowa", transcript = text)
            } else {
                ui.copy(busy = false, status = "❌ Błąd transkrypcji (zobacz Logcat)")
            }
        }
    }

    private fun unloadModels() {
        if (ui.busy) return
        whisperEncoder?.destroy(); whisperDecoder?.destroy()
        whisperEncoder = null; whisperDecoder = null
        ui = ui.copy(loaded = false, status = "Modele zwolnione", transcript = null)
    }

    // ------------- Inference -------------

    /** PCM -> mel -> encodeAuto -> greedy decode -> tekst. */
    private fun transcribeOnce(pcm: FloatArray): String? {
        if (whisperEncoder == null || whisperDecoder == null) return null

        // 1) log-mel [80,3000] (flatten kolumnowo)
        val mel803000 = mel.logMel(pcm)

        // 2) encoder – adaptacyjne dopasowanie kształtu
        val encOut = encodeAuto(mel803000) ?: run {
            Log.e("MainActivity", "encodeAuto: nie udało się dopasować sygnatury encodera")
            return null
        }

        // 3) greedy decode
        val dec = whisperDecoder!!
        val maxSteps = 224
        val prompt = tokenizer.promptIdsAuto().toMutableList()
        val generated = ArrayList<Int>(256)

        // >>> KLUCZOWA ZMIANA: int32 zamiast int64 <<<
        fun idsToTensor(ids: IntArray): Tensor {
            return Tensor.fromBlob(ids, longArrayOf(1L, ids.size.toLong()))
        }

        var tokens = prompt.toIntArray()

        repeat(maxSteps) {
            val ids = idsToTensor(tokens)

            val outs = try {
                // (input_ids:int32, encoder_hidden:float32)
                dec.forward(EValue.from(ids), EValue.from(encOut))
            } catch (e1: Throwable) {
                // awaryjnie odwróć kolejność (nie powinno się uruchomić po fixie)
                try {
                    dec.forward(EValue.from(encOut), EValue.from(ids))
                } catch (e2: Throwable) {
                    Log.e("MainActivity", "decoder forward err:\n1) ${e1.message}\n2) ${e2.message}")
                    return null
                }
            }

            val logits = try { outs.firstOrNull()?.toTensor() } catch (_: Throwable) { null } ?: return null
            val shape = logits.shape() // [1, T, V] albo [1, V]
            val vocab = shape.last().toInt()
            val flat = logits.getDataAsFloatArray()

            val start = if (shape.size == 3) (shape[1].toInt() - 1).coerceAtLeast(0) * vocab else 0
            var bestId = 0
            var best = Float.NEGATIVE_INFINITY
            for (i in 0 until vocab) {
                val v = flat[start + i]
                if (v > best) { best = v; bestId = i }
            }

            generated.add(bestId)
            if (tokenizer.isSpecial(bestId)) return@repeat

            val next = IntArray(tokens.size + 1)
            System.arraycopy(tokens, 0, next, 0, tokens.size)
            next[next.lastIndex] = bestId
            tokens = next
        }

        return tokenizer.decode(generated)
    }

    // -------- Encoder auto-shape helpers --------

    private fun eTensor(data: FloatArray, shape: LongArray) =
        EValue.from(Tensor.fromBlob(data, shape))

    private fun mel80xT_to_Tx80(mel80x3000: FloatArray, T: Int): FloatArray {
        val out = FloatArray(T * 80)
        val srcT = 3000
        val copyT = minOf(T, srcT)
        for (t in 0 until copyT) {
            val dst = t * 80
            for (m in 0 until 80) out[dst + m] = mel80x3000[m * srcT + t]
        }
        return out
    }

    private fun mel80x3000_cropTo80xT(mel80x3000: FloatArray, T: Int): FloatArray {
        val out = FloatArray(80 * T) { 0f }
        val srcT = 3000
        val copyT = minOf(T, srcT)
        for (m in 0 until 80) {
            val dstBase = m * T
            val srcBase = m * srcT
            System.arraycopy(mel80x3000, srcBase, out, dstBase, copyT)
        }
        return out
    }

    private fun tryEncForward(enc: Module, inputs: Array<EValue>): Pair<Tensor?, String?> {
        return try {
            val outs = enc.forward(*inputs)
            val t = outs.firstOrNull()?.toTensor()
            if (t != null) t to null else null to "empty output"
        } catch (e: Throwable) {
            null to "${e.javaClass.simpleName}: ${e.message}"
        }
    }

    private fun encodeAuto(mel80x3000: FloatArray): Tensor? {
        var enc: Module? = whisperEncoder ?: return null
        val T256 = 256

        data class Spec(val desc: String, val inputs: () -> Array<EValue>)
        val specs = mutableListOf<Spec>()

        // 2D, T=3000
        specs += Spec("80x3000") {
            val T = 3000
            arrayOf(eTensor(mel80x3000, longArrayOf(80, T.toLong())))
        }
        specs += Spec("3000x80") {
            val T = 3000
            val data = mel80xT_to_Tx80(mel80x3000, T)
            arrayOf(eTensor(data, longArrayOf(T.toLong(), 80)))
        }
        // 3D (batch=1), T=3000
        specs += Spec("1x80x3000") {
            val T = 3000
            arrayOf(eTensor(mel80x3000, longArrayOf(1, 80, T.toLong())))
        }
        specs += Spec("1x3000x80") {
            val T = 3000
            val data = mel80xT_to_Tx80(mel80x3000, T)
            arrayOf(eTensor(data, longArrayOf(1, T.toLong(), 80)))
        }
        // 2D/3D, T=256
        specs += Spec("80x256") {
            val data = mel80x3000_cropTo80xT(mel80x3000, T256)
            arrayOf(eTensor(data, longArrayOf(80, T256.toLong())))
        }
        specs += Spec("256x80") {
            val data = mel80xT_to_Tx80(mel80x3000, T256)
            arrayOf(eTensor(data, longArrayOf(T256.toLong(), 80)))
        }
        specs += Spec("1x80x256") {
            val data = mel80x3000_cropTo80xT(mel80x3000, T256)
            arrayOf(eTensor(data, longArrayOf(1, 80, T256.toLong())))
        }
        specs += Spec("1x256x80") {
            val data = mel80xT_to_Tx80(mel80x3000, T256)
            arrayOf(eTensor(data, longArrayOf(1, T256.toLong(), 80)))
        }
        // + długość
        specs += Spec("80x3000 + len") {
            val T = 3000
            arrayOf(eTensor(mel80x3000, longArrayOf(80, T.toLong())), EValue.from(T.toLong()))
        }
        specs += Spec("3000x80 + len") {
            val T = 3000
            val data = mel80xT_to_Tx80(mel80x3000, T)
            arrayOf(eTensor(data, longArrayOf(T.toLong(), 80)), EValue.from(T.toLong()))
        }
        specs += Spec("80x256 + len") {
            val T = T256
            val data = mel80x3000_cropTo80xT(mel80x3000, T)
            arrayOf(eTensor(data, longArrayOf(80, T.toLong())), EValue.from(T.toLong()))
        }
        specs += Spec("256x80 + len") {
            val T = T256
            val data = mel80xT_to_Tx80(mel80x3000, T)
            arrayOf(eTensor(data, longArrayOf(T.toLong(), 80)), EValue.from(T.toLong()))
        }

        for (spec in specs) {
            if (enc == null) break
            val (t, err) = tryEncForward(enc!!, spec.inputs())
            if (t != null) {
                Log.i("MainActivity", "encodeAuto OK: ${spec.desc} -> ${t.shape().contentToString()}")
                return t
            } else {
                val msg = err ?: "unknown error"
                Log.w("MainActivity", "encodeAuto FAIL ${spec.desc}: $msg")
                if (msg.contains("Inputs can not be set mid execution")) {
                    runCatching { enc?.destroy() }
                    enc = loader.loadModuleFromAssets(ENCODER_ASSET)
                    val (t2, err2) = if (enc != null) tryEncForward(enc!!, spec.inputs()) else (null to "reload failed")
                    if (t2 != null) {
                        whisperEncoder = enc
                        Log.i("MainActivity", "encodeAuto OK (po reload): ${spec.desc}")
                        return t2
                    } else {
                        Log.w("MainActivity", "encodeAuto retry FAIL ${spec.desc}: $err2")
                    }
                }
            }
        }
        return null
    }
}

// --------------- UI ------------------

private data class UiState(
    val loaded: Boolean = false,
    val busy: Boolean = false,
    val recording: Boolean = false,
    val status: String = "Gotowy",
    val transcript: String? = null,
    val lastError: String? = null
)

@Composable
private fun App(
    state: UiState,
    onPerms: () -> Unit,
    onLoad: () -> Unit,
    onStartRec: () -> Unit,
    onStopTranscribe: () -> Unit,
    onUnload: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text("PTE Whisper (ExecuTorch)", style = MaterialTheme.typography.headlineSmall)

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedButton(onClick = onPerms) { Text("Mikrofon") }
            Button(onClick = onLoad, enabled = !state.loaded && !state.busy) { Text("Załaduj modele") }
            if (!state.recording)
                Button(onClick = onStartRec, enabled = state.loaded && !state.busy) { Text("Start 🎙️") }
            else
                Button(onClick = onStopTranscribe, enabled = state.loaded && !state.busy) { Text("Stop → Transkrybuj") }
            OutlinedButton(onClick = onUnload, enabled = state.loaded && !state.busy) { Text("Unload") }
        }

        if (state.busy) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())

        Text("Status: ${state.status}", maxLines = 2, overflow = TextOverflow.Ellipsis)
        state.transcript?.let {
            Text("Transkrypcja:", style = MaterialTheme.typography.titleMedium)
            Text(it)
        }
        state.lastError?.let { Text("Błąd: $it") }
    }
}

@Preview(showBackground = true, widthDp = 420)
@Composable
private fun PreviewApp() {
    PTEWhisperTheme {
        App(
            state = UiState(),
            onPerms = {},
            onLoad = {},
            onStartRec = {},
            onStopTranscribe = {},
            onUnload = {}
        )
    }
}
