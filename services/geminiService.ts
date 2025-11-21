import { GoogleGenAI } from "@google/genai";
import { ExecutionResult } from '../types';

// Initialize the Gemini client
// Note: API key must be available in process.env.API_KEY
const apiKey = process.env.API_KEY;
let ai: GoogleGenAI | null = null;

if (apiKey) {
  ai = new GoogleGenAI({ apiKey });
}

export const executePythonCode = async (code: string): Promise<ExecutionResult> => {
  if (!ai) {
    return {
      output: "错误: 未在环境变量中找到 API Key。无法运行模拟环境。",
      isError: true
    };
  }

  const modelId = 'gemini-2.5-flash';
  const systemInstruction = `You are a Python Code Execution Simulator. 
  Your goal is to act exactly like a standard Python interpreter.
  
  Rules:
  1. Receive Python code from the user.
  2. Mentally execute the code step-by-step.
  3. Return ONLY the standard output (stdout) that the code would produce.
  4. If the code calculates a value but doesn't print it, do not output it (unless it's the last line of a repl, but assume script mode implies explicit print).
  5. If there is a syntax error or runtime error, simulate the Python error message (traceback) as realistically as possible.
  6. Do not add any markdown formatting (no \`\`\`), no introductory text, no explanations. Just the raw output string.
  `;

  try {
    const response = await ai.models.generateContent({
      model: modelId,
      contents: code,
      config: {
        systemInstruction: systemInstruction,
        temperature: 0.1, // Low temperature for deterministic execution simulation
      }
    });

    const text = response.text || "";
    
    // Simple heuristic to detect error tracebacks for UI styling
    const isError = text.toLowerCase().includes("traceback") || text.toLowerCase().includes("error:");

    return {
      output: text.trim(),
      isError: isError
    };

  } catch (error) {
    console.error("Gemini execution error:", error);
    return {
      output: "系统错误: 无法连接到执行引擎，请稍后再试。",
      isError: true
    };
  }
};