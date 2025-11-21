import React, { useState, useCallback } from 'react';
import { Play, Terminal, Loader2, Trash2, RefreshCw } from 'lucide-react';
import { executePythonCode } from '../services/geminiService';

interface PlaygroundProps {
  initialCode?: string;
}

export const Playground: React.FC<PlaygroundProps> = ({ initialCode = '' }) => {
  
  // Note: Since we are using Gemini to simulate Python, we should actually default to Python code syntax.
  const defaultPythonCode = `# --- Python 交互式演练场 ---
# 你可以在这里编写任何 Python 代码进行实验
# 注意：这是沙箱环境，无法联网或访问本地文件

score = 85

if score >= 60:
    result = "及格 (Pass)"
else:
    result = "不及格 (Fail)"

print(f"分数: {score}, 结果: {result}")

# 尝试一个简单的循环
print("\\n正在计算 1 到 5 的平方:")
for i in range(1, 6):
    print(f"{i} 的平方是 {i**2}")`;

  const [code, setCode] = useState<string>(initialCode || defaultPythonCode);
  const [output, setOutput] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isError, setIsError] = useState(false);

  const handleRun = useCallback(async () => {
    setIsRunning(true);
    setOutput(null); 
    
    try {
      const result = await executePythonCode(code);
      setOutput(result.output);
      setIsError(result.isError);
    } catch (e) {
      setOutput("执行过程中发生意外错误。");
      setIsError(true);
    } finally {
      setIsRunning(false);
    }
  }, [code]);

  const handleReset = () => {
    setCode(defaultPythonCode);
    setOutput(null);
    setIsError(false);
  };

  const handleClearConsole = () => {
    setOutput(null);
  }

  return (
    <div className="rounded-lg overflow-hidden bg-slate-950 shadow-2xl border border-slate-800">
      {/* Toolbar */}
      <div className="bg-slate-900 px-4 py-2.5 border-b border-slate-800 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-slate-400">
            <span className="text-green-500 font-mono font-bold text-lg">›_</span>
            <span className="text-xs font-bold tracking-wide uppercase">Python 3.12 内核</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
           <button 
            onClick={handleReset}
            className="p-1.5 text-slate-500 hover:text-slate-300 transition-colors"
            title="重置代码"
          >
            <RefreshCw size={14} />
          </button>
           <button 
            onClick={handleClearConsole}
            className="p-1.5 text-slate-500 hover:text-slate-300 transition-colors"
            title="清空控制台"
          >
            <Trash2 size={14} />
          </button>
          <div className="h-4 w-px bg-slate-700 mx-1"></div>
          <button
            onClick={handleRun}
            disabled={isRunning}
            className={`
              flex items-center gap-2 px-4 py-1.5 rounded text-xs font-bold transition-all uppercase tracking-wide
              ${isRunning 
                ? 'bg-slate-800 text-slate-500 cursor-wait' 
                : 'bg-green-600 hover:bg-green-500 text-white shadow-[0_0_10px_rgba(22,163,74,0.4)]'
              }
            `}
          >
            {isRunning ? <Loader2 className="animate-spin" size={14} /> : <Play size={14} fill="currentColor" />}
            {isRunning ? '运行中...' : '运行代码'}
          </button>
        </div>
      </div>

      {/* Editor and Console Container */}
      <div className="grid grid-cols-1 lg:grid-cols-2 min-h-[400px]">
        {/* Input Area */}
        <div className="relative border-b lg:border-b-0 lg:border-r border-slate-800">
          <textarea
            value={code}
            onChange={(e) => setCode(e.target.value)}
            className="w-full h-full p-4 bg-[#0d1117] text-slate-300 font-mono text-sm resize-none focus:outline-none leading-relaxed selection:bg-slate-700"
            spellCheck="false"
            placeholder="在此输入 Python 代码..."
          />
        </div>

        {/* Output Area */}
        <div className="flex flex-col bg-[#020408] relative">
          <div className="absolute top-0 left-0 right-0 px-4 py-1 bg-slate-900/50 border-b border-slate-800/50 text-[10px] font-bold text-slate-500 uppercase tracking-wider z-10">
            控制台输出 (Console)
          </div>
          <div className="p-4 pt-8 font-mono text-sm flex-grow overflow-auto h-full custom-scrollbar">
            {isRunning ? (
              <div className="flex items-center gap-2 text-slate-500 italic animate-pulse mt-2">
                <Terminal size={14} />
                正在执行代码...
              </div>
            ) : output !== null ? (
              <div className="animate-in fade-in duration-200">
                  <pre className={`whitespace-pre-wrap break-words font-mono ${isError ? 'text-red-400' : 'text-slate-300'}`}>
                    {isError && <span className="text-red-500 font-bold block mb-1">运行时错误 (RUNTIME ERROR):</span>}
                    {output || <span className="text-slate-600 italic opacity-50">程序执行成功 (无输出)。</span>}
                  </pre>
                  {!isError && output && <div className="mt-4 text-green-500/30 text-xs">Process finished with exit code 0</div>}
              </div>
            ) : (
              <div className="text-slate-700 italic mt-2 select-none">
                // 等待运行...
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};