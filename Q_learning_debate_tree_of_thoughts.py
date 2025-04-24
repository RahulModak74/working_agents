import requests
import json
import sys
import re
import heapq
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import uuid

OLLAMA_API = "http://localhost:11435/api/generate"
MEMORY_FILE = "tot_q_memory.json"

@dataclass
class Thought:
    """Represents a single thought/approach in the tree."""
    id: str
    content: str
    parent_id: Optional[str]
    depth: int
    score: float = 0.0
    children: List[str] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "score": self.score,
            "children": self.children
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Thought':
        return cls(
            id=data["id"],
            content=data["content"],
            parent_id=data["parent_id"],
            depth=data["depth"],
            score=data["score"],
            children=data["children"]
        )

class TreeOfThoughtsAgent:
    def __init__(self, model_name="deepseek-r1", learning_rate=0.7, discount_factor=0.9, 
                 max_depth=3, branching_factor=2, max_attempts=3, api_url=None, api_key=None):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.max_attempts = max_attempts
        self.api_url = api_url or OLLAMA_API
        self.api_key = api_key
        self.q_table = self._load_memory()
        self.thought_tree = {}  # Maps thought_id to Thought objects
        
    def _load_memory(self) -> Dict:
        """Load previous Q-learning memory if it exists."""
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
            
    def _save_memory(self) -> None:
        """Save Q-learning memory to file."""
        with open(MEMORY_FILE, "w") as f:
            json.dump(self.q_table, f)
    
    def _generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        if "ollama" in self.api_url.lower():
            # Ollama API
            response = requests.post(self.api_url, json={
                "model": self.model_name, 
                "prompt": prompt, 
                "stream": False
            }, headers=headers)
            return response.json().get("response", "")
        else:
            # OpenRouter or similar API
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2048
            }, headers=headers)
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
    
    def _evaluate_code_solution(self, solution: str, criteria: List[str]) -> float:
        """Evaluate the quality of a code solution based on multiple criteria."""
        # Extract code blocks
        code_blocks = re.findall(r'```(?:\w+)?\s*([\s\S]*?)```', solution)
        code_content = "\n".join(code_blocks) if code_blocks else solution
        
        score = 0.0
        
        # Check for presence of expected keywords/concepts
        keyword_match = sum(1 for keyword in criteria if keyword.lower() in solution.lower()) / max(len(criteria), 1)
        score += keyword_match * 0.3
        
        # Check for code completeness and structure
        if code_blocks:
            score += 0.2
            
            # Check for function definitions
            if re.search(r'def\s+\w+\s*\(', code_content) or re.search(r'class\s+\w+', code_content):
                score += 0.1
                
            # Check for error handling
            if "try" in code_content and "except" in code_content:
                score += 0.1
                
            # Check for comments/documentation
            if re.search(r'#.*|""".*?"""|\'\'\'.*?\'\'\'', code_content, re.DOTALL):
                score += 0.1
        
        # Check for reasoning steps
        reasoning_patterns = ["step", "first", "second", "third", "then", "next", "finally", "because", "therefore"]
        reasoning_score = sum(1 for pattern in reasoning_patterns if pattern in solution.lower()) / len(reasoning_patterns)
        score += reasoning_score * 0.2
        
        return min(score, 1.0)  # Cap score at 1.0
    
    def _create_thought(self, content: str, parent_id: Optional[str] = None, depth: int = 0) -> Thought:
        """Create a new thought node in the tree."""
        thought_id = str(uuid.uuid4())
        thought = Thought(
            id=thought_id,
            content=content,
            parent_id=parent_id,
            depth=depth
        )
        self.thought_tree[thought_id] = thought
        
        # Update parent's children list if this has a parent
        if parent_id and parent_id in self.thought_tree:
            self.thought_tree[parent_id].children.append(thought_id)
            
        return thought
    
    def _generate_initial_thoughts(self, user_prompt: str) -> List[Thought]:
        """Generate initial approaches to the problem."""
        thoughts = []
        
        initial_prompt = f"""I need to solve this coding problem: {user_prompt}
        
        Generate {self.branching_factor} different initial approaches to solving this problem.
        For each approach:
        1. First explain the high-level strategy
        2. Outline the key steps
        3. Discuss potential advantages and disadvantages
        
        Format each approach with '### Approach X:' as a heading.
        """
        
        response = self._generate_response(initial_prompt)
        
        # Extract different approaches
        approaches = re.split(r'###\s*Approach\s*\d+:', response)
        approaches = [a.strip() for a in approaches if a.strip()]
        
        # Create thoughts for each approach
        for i, approach_content in enumerate(approaches[:self.branching_factor]):
            thought = self._create_thought(content=approach_content, depth=0)
            thoughts.append(thought)
            
        return thoughts
    
    def _expand_thought(self, thought: Thought) -> List[Thought]:
        """Expand a thought by generating more detailed implementations."""
        if thought.depth >= self.max_depth:
            return []
            
        expand_prompt = f"""Based on this approach to solving a coding problem:
        
        {thought.content}
        
        Generate {self.branching_factor} different implementations or refinements of this approach.
        For each implementation:
        1. Show the actual code
        2. Explain key design decisions
        3. Discuss optimizations or improvements
        
        Make sure to include complete, runnable code for each implementation.
        Format each implementation with '### Implementation X:' as a heading.
        """
        
        response = self._generate_response(expand_prompt)
        
        # Extract different implementations
        implementations = re.split(r'###\s*Implementation\s*\d+:', response)
        implementations = [impl.strip() for impl in implementations if impl.strip()]
        
        # Create child thoughts
        child_thoughts = []
        for impl_content in implementations[:self.branching_factor]:
            child = self._create_thought(
                content=impl_content,
                parent_id=thought.id,
                depth=thought.depth + 1
            )
            child_thoughts.append(child)
            
        return child_thoughts
    
    def _critique_thought(self, thought: Thought) -> str:
        """Generate a critique of the thought to find weaknesses."""
        critique_prompt = f"""As a senior developer and code reviewer, critique this implementation:
        
        {thought.content}
        
        Focus on:
        1. Correctness - Will it work as expected?
        2. Efficiency - Are there performance concerns?
        3. Readability - Is the code clear and maintainable?
        4. Error handling - How robust is it?
        5. Edge cases - What might be missing?
        
        Be specific and constructive in your critique.
        """
        
        return self._generate_response(critique_prompt)
    
    def _improve_thought(self, thought: Thought, critique: str) -> str:
        """Improve a thought based on critique."""
        improve_prompt = f"""Here is a code implementation:
        
        {thought.content}
        
        And here is a critique of this implementation:
        
        {critique}
        
        Based on this critique, provide an improved version of the implementation that addresses the issues raised.
        Include the complete code with all improvements.
        """
        
        return self._generate_response(improve_prompt)
    
    def _debate_thoughts(self, thoughts: List[Thought]) -> str:
        """Generate a debate between multiple thoughts to compare their merits."""
        if not thoughts or len(thoughts) < 2:
            return ""
            
        debate_prompt = "I have multiple implementations for a coding problem:\n\n"
        
        for i, thought in enumerate(thoughts):
            debate_prompt += f"Implementation {i+1}:\n{thought.content}\n\n"
            
        debate_prompt += """
        Compare these implementations by discussing:
        1. The strengths and weaknesses of each approach
        2. Which implementation handles edge cases better
        3. Which has better performance characteristics
        4. Which is more maintainable and why
        
        Then synthesize the best elements from all implementations into a final recommended approach.
        """
        
        return self._generate_response(debate_prompt)
    
    def _synthesize_solution(self, best_thoughts: List[Thought], debate_result: str) -> str:
        """Synthesize a final solution based on the best thoughts and debate outcome."""
        synthesis_prompt = f"""Based on these promising implementations:
        
        {"".join(f'Implementation {i+1}:\n{t.content}\n\n' for i, t in enumerate(best_thoughts))}
        
        And this analysis of their strengths and weaknesses:
        
        {debate_result}
        
        Synthesize a final solution that:
        1. Incorporates the strengths of each implementation
        2. Addresses the weaknesses identified
        3. Provides complete, well-documented, and robust code
        4. Includes thoughtful error handling and edge case management
        
        The final solution should be the best possible approach to the problem.
        """
        
        return self._generate_response(synthesis_prompt)
    
    def _extract_criteria(self, prompt: str) -> List[str]:
        """Extract key criteria and concepts from the user's prompt."""
        # Extract programming-related terms
        programming_terms = ["function", "class", "method", "array", "list", "dictionary", 
                            "hash", "tree", "graph", "algorithm", "recursion", "iteration",
                            "optimization", "complexity", "search", "sort", "data structure"]
        
        # Filter out common words
        common_words = {"the", "a", "an", "in", "on", "at", "to", "for", "with", "by", 
                       "as", "of", "and", "or", "but", "if", "then", "than", "that", "this"}
        
        # Extract words and phrases
        words = [word.lower() for word in re.findall(r'\b\w+\b', prompt)]
        important_words = [word for word in words if word not in common_words and len(word) > 3]
        
        # Add any programming terms found in the prompt
        found_terms = [term for term in programming_terms if term.lower() in prompt.lower()]
        
        # Extract specific requirements or constraints
        requirements = re.findall(r'should\s+\w+|must\s+\w+|needs?\s+to\s+\w+', prompt.lower())
        
        return list(set(important_words + found_terms + requirements))
    
    def _get_path_to_root(self, thought_id: str) -> List[Thought]:
        """Get the path from a thought back to the root."""
        path = []
        current_id = thought_id
        
        while current_id is not None:
            if current_id in self.thought_tree:
                thought = self.thought_tree[current_id]
                path.append(thought)
                current_id = thought.parent_id
            else:
                break
                
        return list(reversed(path))
    
    def solve_problem(self, user_prompt: str) -> str:
        """Main method to solve a problem using Tree of Thoughts with debate."""
        attempts = 0
        criteria = self._extract_criteria(user_prompt)
        best_solution = ""
        best_score = 0.0
        
        while attempts < self.max_attempts:
            print(f"🌳 Attempt {attempts+1}/{self.max_attempts}: Growing Tree of Thoughts...")
            
            # Reset thought tree for this attempt
            self.thought_tree = {}
            
            # Generate initial thoughts/approaches
            print("🤔 Generating initial approaches...")
            initial_thoughts = self._generate_initial_thoughts(user_prompt)
            
            # Expand promising thoughts to create a tree
            frontier = [(0.0, thought.id) for thought in initial_thoughts]  # (priority, thought_id)
            heapq.heapify(frontier)
            
            explored_thoughts = 0
            max_explorations = self.max_depth * self.branching_factor * 2  # Limit total explorations
            
            while frontier and explored_thoughts < max_explorations:
                # Get the highest priority thought to expand
                _, thought_id = heapq.heappop(frontier)
                thought = self.thought_tree[thought_id]
                
                print(f"🔍 Exploring thought at depth {thought.depth}...")
                
                # If we're at max depth, evaluate but don't expand
                if thought.depth >= self.max_depth:
                    score = self._evaluate_code_solution(thought.content, criteria)
                    thought.score = score
                    continue
                
                # Critique the current thought
                critique = self._critique_thought(thought)
                
                # Improve the thought based on critique
                improved_content = self._improve_thought(thought, critique)
                
                # Create an improved version of this thought
                improved_thought = self._create_thought(
                    content=improved_content,
                    parent_id=thought.id,
                    depth=thought.depth + 1
                )
                
                # Evaluate the improved thought
                improved_score = self._evaluate_code_solution(improved_content, criteria)
                improved_thought.score = improved_score
                
                # Expand the thought to generate alternatives
                child_thoughts = self._expand_thought(thought)
                
                # Evaluate and add child thoughts to frontier
                for child in child_thoughts:
                    score = self._evaluate_code_solution(child.content, criteria)
                    child.score = score
                    
                    # Use negative score for min-heap to act as max-heap
                    heapq.heappush(frontier, (-score, child.id))
                
                explored_thoughts += 1
            
            # Find the best leaves in the tree
            best_leaves = []
            for thought_id, thought in self.thought_tree.items():
                if not thought.children:  # It's a leaf
                    best_leaves.append(thought)
            
            # Sort leaves by score and take top 3
            best_leaves.sort(key=lambda t: t.score, reverse=True)
            best_leaves = best_leaves[:3]
            
            if best_leaves:
                print(f"🏆 Found {len(best_leaves)} promising solutions, debating merits...")
                
                # Debate the best leaves
                debate_result = self._debate_thoughts(best_leaves)
                
                # Synthesize final solution
                print("🧠 Synthesizing final solution...")
                final_solution = self._synthesize_solution(best_leaves, debate_result)
                
                # Evaluate the final solution
                solution_score = self._evaluate_code_solution(final_solution, criteria)
                
                # Update Q-table for this attempt's approach
                state_key = f"attempt_{attempts}_depth_{self.max_depth}_branch_{self.branching_factor}"
                self.q_table[state_key] = self.q_table.get(state_key, 0) + \
                    self.learning_rate * (solution_score + \
                    self.discount_factor * max(self.q_table.values(), default=0) - \
                    self.q_table.get(state_key, 0))
                
                # Save updated Q-table
                self._save_memory()
                
                print(f"📊 Solution quality score: {solution_score:.2f}")
                
                # Keep track of best solution across attempts
                if solution_score > best_score:
                    best_score = solution_score
                    best_solution = final_solution
                
                # If solution is good enough, return it
                if solution_score > 0.8:
                    print(f"✅ High-quality solution found (score: {solution_score:.2f})")
                    return final_solution
            
            attempts += 1
            
        print(f"⚠️ Max attempts reached. Returning best solution found (score: {best_score:.2f})")
        return best_solution or "Failed to generate a satisfactory solution."
        
    def visualize_tree(self) -> str:
        """Visualize the thought tree in a text-based format."""
        if not self.thought_tree:
            return "Empty thought tree"
            
        # Find root nodes (those with no parents)
        roots = [t for t in self.thought_tree.values() if t.parent_id is None]
        
        result = []
        
        def print_node(node, depth=0):
            indent = "  " * depth
            score_display = f"[Score: {node.score:.2f}]"
            result.append(f"{indent}Thought {node.id[:6]}... {score_display}")
            
            # Show abbreviated content
            content_preview = node.content[:100] + "..." if len(node.content) > 100 else node.content
            content_lines = content_preview.split("\n")
            for line in content_lines[:3]:
                result.append(f"{indent}  {line}")
            if len(content_lines) > 3:
                result.append(f"{indent}  ...")
                
            # Process children
            for child_id in node.children:
                if child_id in self.thought_tree:
                    print_node(self.thought_tree[child_id], depth + 1)
        
        for root in roots:
            print_node(root)
            
        return "\n".join(result)

def main():
    # Ensure filename is provided
    if len(sys.argv) < 2:
        print("❌ Error: Please provide a filename as an argument.")
        sys.exit(1)
        
    file_name = sys.argv[1]
    
    # Check for additional args
    max_depth = 3
    branching = 2
    model = "deepseek-r1"
    api_url = OLLAMA_API
    api_key = None
    
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg.startswith("--depth="):
                max_depth = int(arg.split("=")[1])
            elif arg.startswith("--branch="):
                branching = int(arg.split("=")[1])
            elif arg.startswith("--model="):
                model = arg.split("=")[1]
            elif arg.startswith("--api="):
                api_url = arg.split("=")[1]
            elif arg.startswith("--key="):
                api_key = arg.split("=")[1]
    
    # Read prompt from file
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            user_prompt = f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: File '{file_name}' not found.")
        sys.exit(1)
        
    print(f"🚀 Running Tree of Thoughts with depth={max_depth}, branching={branching}, model={model}")
    
    # Create and run Tree of Thoughts agent
    agent = TreeOfThoughtsAgent(
        model_name=model,
        max_depth=max_depth,
        branching_factor=branching,
        api_url=api_url,
        api_key=api_key
    )
    
    solution = agent.solve_problem(user_prompt)
    
    # Print the final solution
    print("\n=== FINAL SOLUTION ===\n")
    print(solution)
    
    # Show tree visualization
    print("\n=== THOUGHT TREE VISUALIZATION ===\n")
    print(agent.visualize_tree())

if __name__ == "__main__":
    main()
