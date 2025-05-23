import os
import time
import json
import re
import numpy as np

class WorldModelMemory:
    """
    World Model Memory Manager for:
    1. Storing mappings between task IDs and world model summaries
    2. Recording action verification and analysis processes, accumulating all issues
    3. Saving memory to JSON files
    4. Storing action plans, error analyses, and suggested corrected plans
    """
    
    def __init__(self, memory_file="worldmodel_memory.json"):
        """
        Initialize the World Model Memory Manager
        
        Args:
            memory_file: Path to the memory storage file
        """
        self.memory_file = memory_file
        self.task_worldmodels = {}          # task_id -> world_model_summary
        self.task_accumulated_issues = {}   # task_id -> accumulated_issues
        self.action_analyses = {}           # task_id -> list of analysis dicts
        self.corrected_plans = {}           # task_id -> corrected plan dict
        
        # Try to load existing memory
        self._load_memory()
   
    def _load_memory(self):
        """Try to load memory from file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.task_worldmodels = data.get("task_worldmodels", {})
                self.task_accumulated_issues = data.get("task_accumulated_issues", {})
                self.action_analyses = data.get("action_analyses", {})
                self.corrected_plans = data.get("corrected_plans", {})
                print(f"Loaded memory from {self.memory_file}")
            except Exception as e:
                print(f"Error loading memory file: {e}")
                self.task_worldmodels = {}
                self.task_accumulated_issues = {}
                self.action_analyses = {}
                self.corrected_plans = {}
   
    def save_memory(self):
        """Persist memory into JSON file"""
        data = {
            "task_worldmodels": self.task_worldmodels,
            "task_accumulated_issues": self.task_accumulated_issues,
            "action_analyses": self.action_analyses,
            "corrected_plans": self.corrected_plans,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving memory file: {e}")
            return False
   
    def save_worldmodel(self, task_id, worldmodel_summary):
        """
        Save mapping between task ID and world model summary
        Returns saved task_id as string.
        """
        tid = str(task_id)
        self.task_worldmodels[tid] = worldmodel_summary
        if tid not in self.task_accumulated_issues:
            self.task_accumulated_issues[tid] = {
                "object_existence": [],
                "object_state": [],
                "action_sequence": [],
                "spatial_relationship": [],
                "object_accessibility": [],
                "precondition_violation": [],
                "object_property": [],
                "other": []
            }
        self.save_memory()
        return tid
   
    def save_action_analysis(self, task_id, action_plan_str, verification_result):
        """
        Record an action plan analysis under a task
        """
        tid = str(task_id)
        if tid not in self.action_analyses:
            self.action_analyses[tid] = []
        entry = {
            "plan": action_plan_str,
            "analysis": verification_result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        self.action_analyses[tid].append(entry)
        self.save_memory()
   
    def save_corrected_plan(self, task_id, world_model_str, original_plan, corrected_plan, verification_result):
        """
        Store a corrected action plan proposal
        """
        tid = str(task_id)
        self.corrected_plans[tid] = {
            "world_model": world_model_str,
            "original_plan": original_plan,
            "corrected_plan": corrected_plan,
            "analysis": verification_result,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        self.save_memory()
   
    def get_worldmodel(self, task_id):
        """Retrieve world model summary for a task"""
        return self.task_worldmodels.get(str(task_id))
   
    def get_enhanced_worldmodel(self, task_id):
        """
        Retrieve world model with accumulated issues appended
        """
        tid = str(task_id)
        base = self.get_worldmodel(task_id)
        if not base:
            return None
        issues = self.task_accumulated_issues.get(tid, {})
        enhanced = base
        if any(issues.values()):
            enhanced += "\n\nACCUMULATED ISSUES:\n"
            for cat, lst in issues.items():
                if lst:
                    enhanced += f"\n{cat.upper()}:\n"
                    for i, it in enumerate(lst, 1):
                        enhanced += f"{i}. {it}\n"
        return enhanced
    def clear_memory_for_task(self, task_id):
        """
        Clear all stored memory entries for a given task ID
        """
        tid = str(task_id)
        self.task_worldmodels.pop(tid, None)
        self.task_accumulated_issues.pop(tid, None)
        self.action_analyses.pop(tid, None)
        self.corrected_plans.pop(tid, None)
        self.save_memory()
    def clear_all_memory(self):
        """
        Clear all stored worldmodel, issues, analyses, and corrected plans
        """
        self.task_worldmodels.clear()
        self.task_accumulated_issues.clear()
        self.action_analyses.clear()
        self.corrected_plans.clear()
        self.save_memory()
# --- Verification Flow and Helpers ---

def create_verification_prompt(world_model_str, action_plan_str):
    """
    Create a structured verification prompt that simulates each action step-by-step.
    """
    prompt = f"""### WORLD MODEL (YAML)
    {world_model_str}

    ### AVAILABLE ACTION PRIMITIVES
    walk <obj>                  # move agent to object location
    find <obj>                  # alias for walk
    open <container>            # open a container
    close <container>           # close a container
    grab <object>               # pick up an object
    switchon <device>           # turn on a device
    putin <object> <container>  # place object into container
    putback <object> <location> # return object to a location

    ### ACTION SCHEMA
    - walk <loc>
    pre: agent.location != <loc>
    post: agent.location = <loc>

    - find <obj>
    pre: agent.location != objects[<obj>].location
    post: agent.location = objects[<obj>].location

    - open <container>
    pre: agent.location == containers[<container>].location and containers[<container>].state == 'closed'
    post: containers[<container>].state = 'open'

    - close <container>
    pre: agent.location == containers[<container>].location and containers[<container>].state == 'open'
    post: containers[<container>].state = 'closed'

    - grab <object>
    pre: agent.location == objects[<object>].location and len(agent.inventory) < inventory_limit
    post: agent.inventory.append(<object>); objects[<object>].location = 'inventory'

    - switchon <device>
    pre: agent.location == containers[<device>].location and containers[<device>].state in ['closed','off']
    post: containers[<device>].state = 'on'

    - putin <object> <container>
    pre: <object> in agent.inventory and containers[<container>].state == 'open' and agent.location == containers[<container>].location
    post: agent.inventory.remove(<object>); objects[<object>].location = <container>

    - putback <object> <location>
    pre: <object> in agent.inventory and agent.location == <location>
    post: agent.inventory.remove(<object>); objects[<object>].location = <location>

    ### TASK
    Simulate each action in the given list in order; upon the first failing step, stop and report:
    STEP <n> FAILED: <action> — <reason>
    If all steps succeed, output: EXECUTABLE

    Actions:
    {action_plan_str}
    """
    return prompt


def extract_verification_result(verification_response):
    """
    Parse the simulation result. Returns dict with keys:
      - is_executable (bool)
      - first_error (str or None)
      - raw_response (str)
    """
    result = {
        "is_executable": False,
        "first_error": None,
        "raw_response": verification_response
    }
    success_patterns = [
        "EXECUTABLE",
        "All steps succeed",
        "All steps succeed, therefore the output is: EXECUTABLE",
        "**EXECUTABLE**"
    ]
    if any(pattern in verification_response for pattern in success_patterns):
        result["is_executable"] = True
    else:
        m = re.search(r"STEP (\d+) FAILED: (.*?) — (.*)", verification_response)
        if m:
            result["first_error"] = f"Step {m.group(1)} '{m.group(2)}' failed: {m.group(3)}"
    return result


def generate_corrected_plan(world_model_str, action_plan_str, verification_result, max_retries=3):
    """
    Generate a corrected action plan given prior analysis
    """
    prompt = f"""
Given the world model:
{world_model_str}

And the original plan:
{action_plan_str}

The verification result was:
{verification_result}

Create a corrected executable action plan following EXACTLY the same format as the original:
- Each action must be on a single line, in the format "[action] <object> (id)" or "[action] <object1> (id1) <object2> (id2)"
- Separate actions with double newlines
- NO explanations, numbering, or additional text
- NO markdown formatting (bold, italic, etc.)
- Output ONLY the corrected action list

Example format:
[walk] <cabinet> (222)

[open] <cabinet> (222)

[grab] <pancake> (334)

[close] <cabinet> (222)
"""
    try:
        from test_openai import get_response
        corrected = get_response(prompt)
    except ImportError:
        try:
            from test_model import chat_with_model
            # corrected = chat_with_model(prompt)
        except ImportError:
            return None
    corrected = re.sub(r'```[^\n]*\n|\n```', '', corrected)
    
    # Extract only valid action lines
    action_pattern = r'\[.+?\] <.+?> \(\d+\)(?:\s+<.+?> \(\d+\))?'
    matches = re.findall(action_pattern, corrected)
    
    if matches:
        # Join valid actions with double newlines
        corrected = '\n\n'.join(match.strip() for match in matches)
        return corrected
    
    # Fallback: minimal processing if pattern matching fails
    lines = [line.strip() for line in corrected.split('\n') if line.strip()]
    return '\n\n'.join(lines)    
    


def verify_action_plan(world_model_str, action_plan_str, memory=None, max_retries=3, model="gpt-35-turbo"):
    """
    Verify if an action plan is executable in the given world model.
    Returns tuple: (verification_response:str, is_executable:bool)
    """
    prompt = create_verification_prompt(world_model_str, action_plan_str)
    retries = 0
    verification_response = None
    is_executable = False
    inventory_limit = 2

    while retries < max_retries and verification_response is None:
        try:
            try:
                from test_openai import get_response
                verification_response = get_response(prompt)
            except ImportError:
                from test_model import chat_with_model
                # verification_response = chat_with_model(prompt)
        except ImportError:
            verification_response = "STEP 1 FAILED: LLM interface not available — Cannot simulate actions."
        try:
            verification_response = verification_response.strip()
        except:
            verification_response = ""
        parsed = extract_verification_result(verification_response)
        is_executable = parsed["is_executable"]

        if not is_executable and memory:
            task_id = str(int(time.time()))
            memory.save_action_analysis(task_id, action_plan_str, verification_response)
            corrected = generate_corrected_plan(world_model_str, action_plan_str, verification_response)
            if corrected:
                memory.save_corrected_plan(task_id, world_model_str, action_plan_str, corrected, verification_response)

        return verification_response, is_executable

    # If we've exhausted retries without a response
    default = "STEP 1 FAILED: Verification retries exhausted — no response from LLM."
    return default, False
