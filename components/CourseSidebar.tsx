import React, { useState } from 'react';
import { CourseNode } from '../types';
import { ChevronRight, ChevronDown, Book, FileText, Folder } from 'lucide-react';

interface CourseSidebarProps {
  nodes: CourseNode[];
  selectedId: string;
  onSelect: (node: CourseNode) => void;
}

const SidebarItem: React.FC<{ 
  node: CourseNode; 
  selectedId: string; 
  onSelect: (node: CourseNode) => void; 
  level: number 
}> = ({ node, selectedId, onSelect, level }) => {
  const [isOpen, setIsOpen] = useState(level < 1); // Open first levels by default
  const isSelected = node.id === selectedId;
  const hasChildren = node.children && node.children.length > 0;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (node.type === 'category') {
      setIsOpen(!isOpen);
    } else {
      onSelect(node);
    }
  };

  // Styling calculation based on depth
  const paddingLeft = level * 12 + 16;
  const isCategory = node.type === 'category';

  return (
    <div className="select-none">
      <div
        onClick={handleClick}
        className={`
          flex items-center py-2.5 pr-3 cursor-pointer transition-all duration-200
          ${isSelected 
            ? 'bg-primary-900/20 text-primary-400 border-r-2 border-primary-500' 
            : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border-r-2 border-transparent'}
        `}
        style={{ paddingLeft: `${paddingLeft}px` }}
      >
        <span className={`mr-2 shrink-0 ${isSelected ? 'text-primary-500' : 'opacity-60'}`}>
            {hasChildren ? (
                isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />
            ) : (
                isCategory ? <Folder size={14} /> : <FileText size={14} />
            )}
        </span>
        <span className={`text-xs tracking-wide ${isCategory ? 'font-bold text-slate-500' : 'font-medium'} ${isSelected ? 'text-primary-400' : ''} truncate`}>
          {node.title}
        </span>
      </div>
      
      {hasChildren && isOpen && (
        <div className="relative">
          {/* Optional vertical guide line for depth */}
          {level > 0 && (
             <div className="absolute left-[22px] top-0 bottom-0 w-px bg-slate-800/50" style={{ left: `${level * 12 + 8}px` }}></div>
          )}
          {node.children!.map(child => (
            <SidebarItem 
              key={child.id} 
              node={child} 
              selectedId={selectedId} 
              onSelect={onSelect} 
              level={level + 1} 
            />
          ))}
        </div>
      )}
    </div>
  );
};

export const CourseSidebar: React.FC<CourseSidebarProps> = ({ nodes, selectedId, onSelect }) => {
  return (
    <nav className="flex flex-col h-full bg-slate-950 border-r border-slate-800 overflow-y-auto scrollbar-thin w-full">
      <div className="p-5 border-b border-slate-800 bg-slate-950 sticky top-0 z-20 shadow-lg shadow-slate-950/50">
        <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-primary-900/20">
                <Book size={16} className="text-white" />
            </div>
            <div>
                <h2 className="text-slate-100 font-bold text-sm tracking-tight">ML MasterClass</h2>
                <p className="text-xs text-slate-500 font-mono">v3.0.0-CN</p>
            </div>
        </div>
      </div>
      
      <div className="py-4">
        {nodes.map(node => (
          <SidebarItem 
            key={node.id} 
            node={node} 
            selectedId={selectedId} 
            onSelect={onSelect} 
            level={0} 
          />
        ))}
      </div>
    </nav>
  );
};