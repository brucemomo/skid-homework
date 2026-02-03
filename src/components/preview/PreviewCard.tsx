import "react-photo-view/dist/react-photo-view.css";
import { Trash2 } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "../ui/card";
import type { FileItem } from "@/store/problems-store";
import { useCallback, useState, type ClipboardEvent } from "react";
import { cn } from "@/lib/utils";
import { useTranslation } from "react-i18next";
import EmptyPreviewList from "./EmptyPreviewList";
import PreviewList from "./PreviewList";

export type PreviewCardProps = {
  items: FileItem[];
  appendFiles: (files: File[] | FileList, source: FileItem["source"]) => void;
  removeItem: (id: string) => void;
  layout?: "default" | "mobile";
};

export default function PreviewCard({
  items,
  removeItem,
  appendFiles,
  layout = "default",
}: PreviewCardProps) {
  const { t } = useTranslation("commons", { keyPrefix: "preview" });

  const [isDragging, setIsDragging] = useState(false);
  const isMobileLayout = layout === "mobile";

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      if (isMobileLayout) return;
      setIsDragging(false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        appendFiles(e.dataTransfer.files, "upload");
      }
    },
    [appendFiles, isMobileLayout],
  );

  const handlePaste = (e: ClipboardEvent) => {
    e.preventDefault();
    if (!e.clipboardData) return;
    appendFiles(e.clipboardData.files, "upload");
  };

  return (
    <>
      <Card
        // contentEditable
        tabIndex={0}
        onPaste={handlePaste}
        suppressContentEditableWarning
        // onKeyDown={preventTyping}
        className={cn(
          "md:col-span-2 border-white/10 backdrop-blur outline-none caret-transparent cursor-default",
          isMobileLayout &&
            "border border-white/20 bg-background/70 shadow-lg backdrop-blur-lg",
        )}
      >
        <CardHeader className={cn(isMobileLayout && "px-5 pb-2 pt-5")}>
          <CardTitle
            className={cn(
              "text-base",
              isMobileLayout && "text-lg font-semibold",
            )}
          >
            {t("title")}
          </CardTitle>
        </CardHeader>
        <CardContent
          className="flex flex-col gap-2"
          onDragOver={(e) => {
            if (isMobileLayout) return;
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => !isMobileLayout && setIsDragging(false)}
        >
          {items.length === 0 ? (
            <EmptyPreviewList
              layout={layout}
              isDragging={isDragging}
              onDrop={onDrop}
            />
          ) : (
            <PreviewList
              isDragging={isDragging}
              onDrop={onDrop}
              layout={layout}
              removeItem={removeItem}
            />
          )}

          {isDragging && !isMobileLayout && (
            <div
              className="flex h-64 flex-col items-center justify-center rounded-lg border border-dashed text-slate-400 border-red-500 bg-red-500/10"
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
              }}
            >
              <Trash2 />
              {t("drop-cancel")}
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}
