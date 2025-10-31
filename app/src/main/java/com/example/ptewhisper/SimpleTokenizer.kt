package com.example.ptewhisper

import android.content.Context
import org.json.JSONObject

class SimpleTokenizer(private val context: Context) {
    private val id2tok: Array<String>
    private val eosId: Int
    private val noTsId: Int
    private val sotId: Int
    private val transcribeId: Int

    init {
        // tokenizer.json – HuggingFace format
        val tokJson = context.assets.open("tokenizer.json").bufferedReader().use { it.readText() }
        val root = JSONObject(tokJson)
        val model = root.getJSONObject("model")
        val vocab = model.getJSONObject("vocab")
        val size = vocab.length()
        val arr = Array(size) { "" }
        val it = vocab.keys()
        while (it.hasNext()) {
            val k = it.next()
            val id = vocab.getInt(k)
            if (id in arr.indices) arr[id] = k
        }
        id2tok = arr

        // config.json – specjalne ID (jeśli są)
        val cfgText = context.assets.open("config.json").bufferedReader().use { it.readText() }
        val cfg = JSONObject(cfgText)
        eosId = cfg.optInt("eos_token_id", 50257)
        sotId = cfg.optInt("sot_token_id", 50258)
        transcribeId = cfg.optInt("transcribe_token_id", 50359)
        noTsId = cfg.optInt("no_timestamps_token_id", 50363)
    }

    fun promptIdsAuto(): IntArray {
        // klasyczny prompt: <|sot|> <|transcribe|> <|no_timestamps|>
        return intArrayOf(sotId, transcribeId, noTsId)
    }

    fun isSpecial(id: Int): Boolean {
        // <|...|> – usuń ze stringa
        val s = if (id in id2tok.indices) id2tok[id] else ""
        return s.startsWith("<|") && s.endsWith("|>")
    }

    fun decode(ids: List<Int>): String {
        val sb = StringBuilder()
        for (id in ids) {
            if (id == eosId) break
            val s = if (id in id2tok.indices) id2tok[id] else ""
            if (isSpecial(id)) continue
            // heurystyka spacji: wiele BPE ma 'Ġ' (GPT2) albo '▁' (SentencePiece)
            val cleaned = s
                .replace("Ġ", " ")
                .replace("▁", " ")
                .replace("ÃĤ", "Ĥ") // czasem artefakty UTF-8 – minimalna sanit.
            sb.append(cleaned)
        }
        return sb.toString().replace(Regex(" +"), " ").trim()
    }
}
