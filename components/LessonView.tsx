import React from 'react';
import { CourseNode } from '../types';
import { Playground } from './Playground';
import { BookOpen, Lightbulb, Terminal } from 'lucide-react';

interface LessonViewProps {
  node: CourseNode;
}

export const LessonView: React.FC<LessonViewProps> = ({ node }) => {
  if (!node.content) return <div className="p-8 text-slate-500 flex items-center justify-center h-full">请选择左侧课程以开始学习。</div>;

  return (
    <div className="max-w-5xl mx-auto p-6 lg:p-10 pb-32">
      {/* Breadcrumb / Header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 text-primary-500 mb-3 text-xs font-bold tracking-widest uppercase">
          <BookOpen size={14} />
          <span>Current Module (当前模块)</span>
        </div>
        <h1 className="text-3xl lg:text-4xl font-bold text-slate-100 mb-4 leading-tight">{node.title}</h1>
        <div className="prose prose-invert prose-slate max-w-none">
            <p className="text-lg text-slate-400 leading-relaxed border-l-4 border-slate-800 pl-4">
            {node.content.description}
            </p>
        </div>
      </div>

      {/* Tags */}
      <div className="flex flex-wrap gap-2 mb-10">
        {node.content.keyConcepts.map(concept => (
          <span key={concept} className="px-2.5 py-1 rounded bg-slate-800/50 border border-slate-700/50 text-slate-400 text-xs font-mono">
            {concept}
          </span>
        ))}
      </div>

      {/* Example Code Section */}
      <div className="space-y-4 mb-12">
        <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-slate-200 flex items-center gap-2">
                示例代码讲解
            </h3>
        </div>
        
        {/* Read-Only Code Window */}
        <div className="rounded-lg overflow-hidden bg-[#0d1117] border border-slate-800 shadow-xl">
            <div className="flex items-center justify-between px-4 py-2 bg-slate-900/50 border-b border-slate-800">
                <div className="flex space-x-2">
                    <div className="w-3 h-3 rounded-full bg-red-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-green-500/80"></div>
                </div>
                <div className="text-xs font-mono text-slate-500 flex items-center gap-2">
                    <span className="opacity-50">example_code.py</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600"></span>
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-600"></span>
                    <span className="text-[10px] text-slate-500 font-bold tracking-wider ml-1">READ ONLY</span>
                </div>
            </div>
            <div className="relative">
                <pre className="p-5 text-sm font-mono text-slate-300 overflow-x-auto leading-relaxed">
                    <code>{node.content.codeExample}</code>
                </pre>
            </div>
        </div>

        {/* Explanation Box */}
        <div className="bg-slate-900/50 border-l-2 border-primary-500/50 p-4 rounded-r flex gap-4">
            <Lightbulb className="text-primary-400 shrink-0 mt-0.5" size={18} />
            <div className="text-slate-400 text-sm leading-relaxed">
                {node.content.explanation}
            </div>
        </div>
      </div>

      {/* Playground Section Header */}
      <div className="flex items-center gap-4 mb-6">
        <div className="h-px bg-slate-800 flex-grow"></div>
        <div className="flex items-center gap-2 text-primary-400">
            <Terminal size={18} />
            <span className="font-bold text-sm tracking-wider uppercase">Interactive Lab (代码演练场)</span>
        </div>
        <div className="h-px bg-slate-800 flex-grow"></div>
      </div>

      {/* The Interactive Playground */}
      <div className="bg-slate-900/30 rounded-xl border border-slate-800/50 p-1">
        <Playground />
      </div>
    </div>
  );
};