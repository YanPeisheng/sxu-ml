import React, { useState, useEffect } from 'react';
import { CourseSidebar } from './components/CourseSidebar';
import { LessonView } from './components/LessonView';
import { COURSE_DATA } from './constants';
import { CourseNode } from './types';
import { Menu, X, Github } from 'lucide-react';

const App: React.FC = () => {
  // State for selected lesson
  const [selectedNode, setSelectedNode] = useState<CourseNode | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true); // Default open on desktop
  const [isMobile, setIsMobile] = useState(false);

  // Select first lesson on load
  useEffect(() => {
    const firstCategory = COURSE_DATA[0];
    if (firstCategory && firstCategory.children && firstCategory.children.length > 0) {
      // Try to select the first actual lesson (grandchild) if possible
      const firstSub = firstCategory.children[0];
      if (firstSub.children && firstSub.children.length > 0) {
          setSelectedNode(firstSub.children[0]);
      } else {
          setSelectedNode(firstSub);
      }
    }

    // Handle responsiveness
    const handleResize = () => {
      if (window.innerWidth < 1024) {
        setIsMobile(true);
        setSidebarOpen(false);
      } else {
        setIsMobile(false);
        setSidebarOpen(true);
      }
    };

    handleResize(); // Initial check
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleNodeSelect = (node: CourseNode) => {
    setSelectedNode(node);
    if (isMobile) {
      setSidebarOpen(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-200 overflow-hidden">
      
      {/* Mobile Overlay */}
      {isMobile && sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/60 z-20 backdrop-blur-sm"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside 
        className={`
          fixed lg:static inset-y-0 left-0 z-30 w-72 transform transition-transform duration-300 ease-in-out border-r border-slate-800 shadow-2xl lg:shadow-none
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        <CourseSidebar 
          nodes={COURSE_DATA} 
          selectedId={selectedNode?.id || ''} 
          onSelect={handleNodeSelect} 
        />
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        
        {/* Mobile Header */}
        <div className="lg:hidden flex items-center justify-between p-4 border-b border-slate-800 bg-slate-900/80 backdrop-blur">
          <button 
            onClick={() => setSidebarOpen(true)}
            className="p-2 -ml-2 text-slate-400 hover:text-white"
          >
            <Menu size={24} />
          </button>
          <span className="font-bold text-slate-200">ML Masterclass</span>
          <div className="w-8" /> {/* Spacer */}
        </div>

        {/* Close button for sidebar on mobile (inside sidebar visual area visually but handled by overlay usually, keeping clean) */}
        {isMobile && sidebarOpen && (
           <button 
             onClick={() => setSidebarOpen(false)}
             className="absolute top-4 left-64 z-40 text-slate-400 p-2"
           >
             <X size={20} />
           </button>
        )}

        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-y-auto scrollbar-thin scroll-smooth">
          {selectedNode ? (
            <LessonView node={selectedNode} />
          ) : (
            <div className="flex items-center justify-center h-full text-slate-500">
              请选择课程开始学习。
            </div>
          )}
          
          {/* Footer */}
          <footer className="py-8 text-center text-slate-600 text-sm border-t border-slate-800/50 mt-auto">
            <div className="flex items-center justify-center gap-2 mb-2">
               <span className="font-semibold text-slate-500">ML MasterClass Studio</span>
            </div>
            <p>&copy; {new Date().getFullYear()} 绍兴文理学院</p>
          </footer>
        </div>

      </main>
    </div>
  );
};

export default App;