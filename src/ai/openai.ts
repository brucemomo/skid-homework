import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import type { AiChatMessage } from "./chat-types";

export type OpenAiModel = {
  name: string;
  displayName: string;
};

const DEFAULT_OPENAI_ROOT = "https://api.openai.com/v1";

function normalizeBaseUrl(baseUrl?: string) {
  const normalized = (baseUrl ?? DEFAULT_OPENAI_ROOT).replace(/\/$/, "");
  return normalized;
}

export class OpenAiClient {
  private client: OpenAI;
  private systemPrompts: string[];
  private toolPrompts: string[];

  constructor(apiKey: string, baseUrl?: string) {
    this.client = new OpenAI({
      apiKey,
      baseURL: normalizeBaseUrl(baseUrl),
      dangerouslyAllowBrowser: true,
    });
    this.systemPrompts = [];
    this.toolPrompts = [];
  }

  addSystemPrompt(prompt: string) {
    this.systemPrompts?.push(prompt);
  }

  setAvailableTools(prompts: string[]) {
    this.toolPrompts = prompts;
  }

  private buildSystemPrompt(): string {
    let prompt = this.systemPrompts.join("\n\n");

    if (this.toolPrompts.length > 0) {
      // build tool calling prompts
      prompt += "\n## Available Tools\n\n";
      prompt += this.toolPrompts.join("\n\n");
    }

    return prompt;
  }

  /**
   * Sends a request with an image (Vision API).
   */
  async sendMedia(
    media: string,
    mimeType: string,
    prompt?: string,
    model = "gpt-4o-mini",
    callback?: (text: string) => void,
  ) {
    const messages: ChatCompletionMessageParam[] = [];

    // 1. Add System Prompt
    if (this.systemPrompts) {
      const systemPrompt = this.buildSystemPrompt();
      messages.push({
        role: "system",
        content: systemPrompt,
      });
    }

    // 2. Build User Content (Text + Image)
    const contentParts: Array<
      | { type: "text"; text: string }
      | {
          type: "image_url";
          image_url: { url: string; detail?: "auto" | "low" | "high" };
        }
    > = [];

    if (prompt) {
      contentParts.push({
        type: "text",
        text: prompt,
      });
    }

    contentParts.push({
      type: "image_url",
      image_url: {
        url: `data:${mimeType};base64,${media}`,
        detail: "auto",
      },
    });

    messages.push({
      role: "user",
      content: contentParts,
    });

    return this._executeStream(model, messages, callback);
  }

  /**
   * Sends a standard text-only chat request.
   */
  async sendChat(
    messages: AiChatMessage[],
    model = "gpt-4o-mini",
    callback?: (text: string) => void,
  ) {
    const openAiMessages: ChatCompletionMessageParam[] = [];

    // 1. Add System Prompt
    if (this.systemPrompts) {
      const systemPrompt = this.buildSystemPrompt();
      messages.push({
        role: "system",
        content: systemPrompt,
      });
    }

    console.log(
      `AI Query with ${model}\nSystem prompt:`,
      this.systemPrompts,
      "\nUser query:",
      messages,
    );

    // 2. Convert History
    for (const message of messages) {
      const trimmed = message.content?.trim();
      if (!trimmed) continue;

      const role =
        message.role === "assistant"
          ? "assistant"
          : message.role === "system"
            ? "system"
            : "user";

      openAiMessages.push({
        role: role,
        content: trimmed,
      });
    }

    return this._executeStream(model, openAiMessages, callback);
  }

  /**
   * Internal helper to handle the streaming response from OpenAI.
   */
  private async _executeStream(
    model: string,
    messages: ChatCompletionMessageParam[],
    callback?: (text: string) => void,
  ): Promise<string> {
    const stream = await this.client.chat.completions.create({
      model,
      messages,
      stream: true,
    });

    let aggregated = "";

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content || "";

      if (delta) {
        aggregated += delta;
        callback?.(delta);
      }
    }

    return aggregated.trim();
  }

  async getAvailableModels(): Promise<OpenAiModel[]> {
    const response = await this.client.models.list();

    return response.data.map((model) => ({
      name: model.id,
      displayName: model.id,
    }));
  }
}
