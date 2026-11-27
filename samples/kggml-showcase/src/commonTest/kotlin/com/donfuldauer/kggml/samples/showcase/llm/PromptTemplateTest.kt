package com.donfuldauer.kggml.samples.showcase.llm

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull

class PromptTemplateTest {
    private val conversation = listOf(
        ChatMessage(ChatRole.USER, "Hi"),
        ChatMessage(ChatRole.ASSISTANT, "Hello!"),
        ChatMessage(ChatRole.USER, "What is this?"),
    )

    @Test
    fun chatMlRendersSystemTurnsAndOpensTheAssistantTurn() {
        assertEquals(
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n" +
                "<|im_start|>user\n<__media__>\nWhat is this?<|im_end|>\n<|im_start|>assistant\n",
            PromptTemplate.CHATML.format("Be brief.", conversation, imageCount = 1),
        )
    }

    @Test
    fun gemmaFoldsTheSystemPromptIntoTheFirstTurn() {
        assertEquals(
            "<start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\nHello!<end_of_turn>\n" +
                "<start_of_turn>user\nWhat is this?<end_of_turn>\n<start_of_turn>model\n",
            PromptTemplate.GEMMA.format("Be brief.", conversation),
        )
    }

    @Test
    fun userAssistantFormatMatchesSmolVlm() {
        assertEquals(
            "<|im_start|>User: <__media__><__media__>\nDescribe<end_of_utterance>\nAssistant:",
            PromptTemplate.USER_ASSISTANT.format("", listOf(ChatMessage(ChatRole.USER, "Describe")), imageCount = 2),
        )
    }

    @Test
    fun forModelRecognizesTheModelFamilyFromItsFileName() {
        assertEquals(PromptTemplate.CHATML, PromptTemplate.forModel("/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"))
        assertEquals(PromptTemplate.CHATML, PromptTemplate.forModel("C:\\models\\qwen2.5-1.5b-instruct.gguf"))
        assertEquals(PromptTemplate.CHATML, PromptTemplate.forModel("SmolLM2-360M-Instruct-Q8_0.gguf"))
        assertEquals(PromptTemplate.USER_ASSISTANT, PromptTemplate.forModel("/sdcard/SmolVLM-256M-Instruct-Q8_0.gguf"))
        assertEquals(PromptTemplate.GEMMA, PromptTemplate.forModel("gemma-3-1b-it-q4_0.gguf"))
        assertNull(PromptTemplate.forModel("Llama-3.2-1B-Instruct-Q4_K_M.gguf"), "unknown family: no guess")
        assertNull(PromptTemplate.forModel(""))
    }

    @Test
    fun cleanStripsStopSequences() {
        assertEquals("Hello", PromptTemplate.CHATML.clean(" Hello<|im_end|>"))
        assertEquals("A cat.", PromptTemplate.USER_ASSISTANT.clean(" A cat.<end_of_utterance>"))
        assertEquals("Yes", PromptTemplate.USER_ASSISTANT.clean("Yes\nUser: more"))
    }
}
