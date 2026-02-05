"use client";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import CodeBlock, { MarkdownCodeProps, SourceContext } from "./CodeBlock";
import { memo } from "react";

const components = {
  code: CodeBlock,
  ol: ({ ...props }: MarkdownCodeProps) => (
    <ol style={{ listStyleType: "decimal", paddingLeft: "20px" }} {...props} />
  ),
  li: ({ ...props }: MarkdownCodeProps) => (
    <li style={{ marginBottom: "4px" }} {...props} />
  ),
};

const MarkdownRenderer = ({ source }: { source: string }) => {
  return (
    <SourceContext.Provider value={source}>
      <Markdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { output: "html" }]]}
        components={components}
      >
        {source}
      </Markdown>
    </SourceContext.Provider>
  );
};

export const MemoizedMarkdown = memo(MarkdownRenderer);
