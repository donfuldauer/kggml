package com.donfuldauer.kggml.samples.showcase.llm

import kotlin.test.Test
import kotlin.test.assertEquals

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
    fun cleanStripsStopSequences() {
        assertEquals("Hello", PromptTemplate.CHATML.clean(" Hello<|im_end|>"))
        assertEquals("A cat.", PromptTemplate.USER_ASSISTANT.clean(" A cat.<end_of_utterance>"))
        assertEquals("Yes", PromptTemplate.USER_ASSISTANT.clean("Yes\nUser: more"))
    }
}
