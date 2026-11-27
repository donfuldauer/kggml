package com.donfuldauer.kggml.samples.showcase.llm

import com.donfuldauer.kggml.llama.LlamaEngine

/** Who wrote a chat message. */
enum class ChatRole { USER, ASSISTANT }

data class ChatMessage(val role: ChatRole, val text: String)

/**
 * Chat prompt formats. The showcase formats prompts itself so that every turn re-renders the whole conversation:
 * the engine then reuses the KV cache of the common prefix and only evaluates the new turn.
 */
enum class PromptTemplate(val label: String, val stopSequences: List<String>) {
    /** Qwen, SmolLM, many fine-tunes (`<|im_start|>role … <|im_end|>`). */
    CHATML("ChatML", listOf("<|im_end|>")),

    /** Gemma (`<start_of_turn>user … <end_of_turn>`). */
    GEMMA("Gemma", listOf("<end_of_turn>")),

    /** SmolVLM / Idefics (`User: … <end_of_utterance>\nAssistant:`). */
    USER_ASSISTANT("User / Assistant", listOf("<end_of_utterance>", "\nUser:")),
    ;

    /**
     * Renders [system] + [messages] and opens the assistant's next turn. [imageCount] media markers
     * ([LlamaEngine.MEDIA_MARKER]) are placed at the start of the last user message.
     */
    fun format(system: String, messages: List<ChatMessage>, imageCount: Int = 0): String = buildString {
        val lastUser = messages.indexOfLast { it.role == ChatRole.USER }
        fun content(index: Int, message: ChatMessage): String =
            if (index == lastUser && imageCount > 0) LlamaEngine.MEDIA_MARKER.repeat(imageCount) + "\n" + message.text else message.text
        when (this@PromptTemplate) {
            CHATML -> {
                if (system.isNotBlank()) append("<|im_start|>system\n$system<|im_end|>\n")
                messages.forEachIndexed { i, m ->
                    val role = if (m.role == ChatRole.USER) "user" else "assistant"
                    append("<|im_start|>$role\n${content(i, m)}<|im_end|>\n")
                }
                append("<|im_start|>assistant\n")
            }

            GEMMA -> {
                messages.forEachIndexed { i, m ->
                    val role = if (m.role == ChatRole.USER) "user" else "model"
                    // Gemma has no system role: the system prompt prefixes the first user turn.
                    val text = if (i == 0 && system.isNotBlank()) "$system\n\n${content(i, m)}" else content(i, m)
                    append("<start_of_turn>$role\n$text<end_of_turn>\n")
                }
                append("<start_of_turn>model\n")
            }

            USER_ASSISTANT -> {
                append("<|im_start|>")
                if (system.isNotBlank()) append("System: $system<end_of_utterance>\n")
                messages.forEachIndexed { i, m ->
                    if (m.role == ChatRole.USER) append("User: ${content(i, m)}<end_of_utterance>\n")
                    else append("Assistant: ${content(i, m)}<end_of_utterance>\n")
                }
                append("Assistant:")
            }
        }
    }

    companion object {
        /**
         * The template a model was trained on, guessed from its file name or path (the GGUF names of the catalog and
         * of most community conversions name the family), or `null` when the name does not tell. A mismatched
         * template makes a model ramble: it never produces the stop sequence of a format it does not know.
         */
        fun forModel(path: String): PromptTemplate? {
            val name = path.substringAfterLast('/').substringAfterLast('\\').lowercase()
            return when {
                name.isBlank() -> null
                "smolvlm" in name || "idefics" in name -> USER_ASSISTANT
                "gemma" in name -> GEMMA
                listOf("qwen", "smollm", "chatml", "hermes").any { it in name } -> CHATML
                else -> null
            }
        }
    }

    /** Removes a trailing stop sequence (the engine emits the token that completed it) and surrounding blanks. */
    fun clean(text: String): String {
        var out = text
        stopSequences.forEach { stop -> out = out.substringBefore(stop) }
        return out.trim()
    }
}
