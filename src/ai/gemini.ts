import { GoogleGenAI, HarmBlockThreshold, HarmCategory } from "@google/genai";
import type { AiChatMessage } from "./chat-types";

export interface GeminiModel {
  name: string;
  displayName: string;
}

export interface GeminiConfig {
  thinkingBudget?: number;
  safetySettings?: Array<{
    category: HarmCategory;
    threshold: HarmBlockThreshold;
  }>;
}

export class GeminiAi {
  private ai: GoogleGenAI;
  private systemPrompts: string[];
  private toolPrompts: string[];
  private config: GeminiConfig;

  constructor(key: string, baseUrl?: string, config?: GeminiConfig) {
    this.ai = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        baseUrl: baseUrl,
      },
    });

    this.systemPrompts = [];
    this.toolPrompts = [];

    this.config = {
      thinkingBudget: config?.thinkingBudget ?? -1,
      safetySettings: config?.safetySettings ?? [
        {
          category: HarmCategory.HARM_CATEGORY_HARASSMENT,
          threshold: HarmBlockThreshold.BLOCK_NONE,
        },
        {
          category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
          threshold: HarmBlockThreshold.BLOCK_NONE,
        },
        {
          category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
          threshold: HarmBlockThreshold.BLOCK_NONE,
        },
        {
          category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
          threshold: HarmBlockThreshold.BLOCK_NONE,
        },
      ],
    };
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

  async sendMedia(
    media: string,
    mimeType: string,
    prompt?: string,
    model = "gemini-2.5-pro",
    callback?: (text: string) => void,
  ) {
    const contents = [];

    if (this.systemPrompts) {
      const systemPrompt = this.buildSystemPrompt();
      contents.push({
        role: "user",
        parts: [{ text: systemPrompt }],
      });
    }

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const parts: any[] = [];
    if (prompt) {
      parts.push({ text: prompt });
    }

    if (media.startsWith("http")) {
      parts.push({
        fileData: {
          mimeType,
          fileUri: media,
        },
      });
    } else {
      parts.push({
        inlineData: {
          mimeType,
          data: media, // base64
        },
      });
    }

    contents.push({
      role: "user",
      parts,
    });

    const response = await this.ai.models.generateContentStream({
      model,
      config: {
        thinkingConfig: { thinkingBudget: this.config.thinkingBudget },
        safetySettings: this.config.safetySettings,
      },
      contents,
    });

    let result = "";
    for await (const chunk of response) {
      if (chunk.text) {
        result += chunk.text;
        callback?.(chunk.text);
      }
    }
    return result;
  }

  async getAvailableModels(): Promise<GeminiModel[]> {
    const models = await this.ai.models.list();
    return models.page.map((it) => ({
      name: it.name!,
      displayName: it.displayName ?? it.name!,
    }));
  }

  async sendChat(
    messages: AiChatMessage[],
    model = "gemini-2.5-pro",
    callback?: (text: string) => void,
  ) {
    const contents = [];

    if (this.systemPrompts) {
      const systemPrompt = this.buildSystemPrompt();
      contents.push({
        role: "user",
        parts: [{ text: systemPrompt }],
      });
    }

    console.log(
      `AI Query with ${model}\nSystem prompt:`,
      this.systemPrompts,
      "\nUser query:",
      messages,
    );

    for (const message of messages) {
      const trimmed = message.content?.trim();
      if (!trimmed) continue;

      const role = message.role === "assistant" ? "model" : "user";

      contents.push({
        role,
        parts: [{ text: trimmed }],
      });
    }

    const response = await this.ai.models.generateContentStream({
      model,
      config: {
        thinkingConfig: { thinkingBudget: this.config.thinkingBudget },
        safetySettings: this.config.safetySettings,
      },
      contents,
    });

    let result = "";
    for await (const chunk of response) {
      if (chunk.text) {
        result += chunk.text;
        callback?.(chunk.text);
      }
    }
    return result.trim();
  }
}
