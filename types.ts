export interface CourseNode {
  id: string;
  title: string;
  type: 'category' | 'lesson';
  children?: CourseNode[];
  content?: LessonContent;
}

export interface LessonContent {
  description: string;
  keyConcepts: string[];
  codeExample: string;
  explanation: string;
}

export interface ExecutionResult {
  output: string;
  isError: boolean;
}