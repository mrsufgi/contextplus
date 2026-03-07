// Dependency graph analyzer to trace symbol usage across the codebase
// Finds every file and line where a function, class, or variable is referenced

import { walkDirectory } from "../core/walker.js";
import { isSupportedFile } from "../core/parser.js";
import { readFile } from "fs/promises";
import { getCachedFileLines } from "./semantic-identifiers.js";

const FILE_CONCURRENCY = 20;

export interface BlastRadiusOptions {
  rootDir: string;
  symbolName: string;
  fileContext?: string;
}

interface SymbolUsage {
  file: string;
  line: number;
  context: string;
}

export async function getBlastRadius(options: BlastRadiusOptions): Promise<string> {
  const cachedLines = getCachedFileLines();
  const usages: SymbolUsage[] = [];
  const symbolPattern = new RegExp(`\\b${escapeRegex(options.symbolName)}\\b`, "g");

  if (cachedLines && cachedLines.size > 0) {
    for (const [relativePath, lines] of cachedLines) {
      for (let i = 0; i < lines.length; i++) {
        if (symbolPattern.test(lines[i])) {
          const isDefinition = options.fileContext && relativePath === options.fileContext && isDefinitionLine(lines[i], options.symbolName);
          if (!isDefinition) {
            usages.push({ file: relativePath, line: i + 1, context: lines[i].trim().substring(0, 120) });
          }
          symbolPattern.lastIndex = 0;
        }
      }
    }
  } else {
    const entries = await walkDirectory({ rootDir: options.rootDir, depthLimit: 0 });
    const files = entries.filter((e) => !e.isDirectory && isSupportedFile(e.path));
    for (let fi = 0; fi < files.length; fi += FILE_CONCURRENCY) {
      const batch = files.slice(fi, fi + FILE_CONCURRENCY);
      const results = await Promise.allSettled(
        batch.map(async (file) => {
          const content = await readFile(file.path, "utf-8");
          return { file, lines: content.split("\n") };
        })
      );
      for (const result of results) {
        if (result.status !== "fulfilled") continue;
        const { file, lines: fileLines } = result.value;
        for (let i = 0; i < fileLines.length; i++) {
          if (symbolPattern.test(fileLines[i])) {
            const isDef = options.fileContext && file.relativePath === options.fileContext && isDefinitionLine(fileLines[i], options.symbolName);
            if (!isDef) {
              usages.push({ file: file.relativePath, line: i + 1, context: fileLines[i].trim().substring(0, 120) });
            }
            symbolPattern.lastIndex = 0;
          }
        }
      }
    }
  }

  if (usages.length === 0) return `Symbol "${options.symbolName}" is not used anywhere in the codebase.`;

  const byFile = new Map<string, SymbolUsage[]>();
  for (const u of usages) {
    const existing = byFile.get(u.file) ?? [];
    existing.push(u);
    byFile.set(u.file, existing);
  }

  const lines: string[] = [
    `Blast radius for "${options.symbolName}": ${usages.length} usages in ${byFile.size} files\n`,
  ];

  for (const [file, fileUsages] of byFile) {
    lines.push(`  ${file}:`);
    for (const u of fileUsages) {
      lines.push(`    L${u.line}: ${u.context}`);
    }
  }

  if (usages.length <= 1) {
    lines.push(`\n⚠ LOW USAGE: This symbol is used only ${usages.length} time(s). Consider inlining if it's under 20 lines.`);
  }

  return lines.join("\n");
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const DEF_PATTERN_1 = /(?:function|class|enum|interface|struct|type|trait|fn|def|func)\s+/;
const DEF_PATTERN_2 = /(?:const|let|var|pub|export)\s+(?:async\s+)?(?:function\s+)?/;

function isDefinitionLine(line: string, symbolName: string): boolean {
  const idx1 = line.search(DEF_PATTERN_1);
  if (idx1 !== -1 && line.includes(symbolName, idx1)) return true;
  const idx2 = line.search(DEF_PATTERN_2);
  if (idx2 !== -1 && line.includes(symbolName, idx2)) return true;
  return false;
}
