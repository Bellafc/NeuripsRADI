import os
import time
import json
import re
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class WorldModelMemory:
    """
    World Model Memory Manager with vector-based retrieval capabilities:
    1. Storing mappings between task IDs and world model summaries
    2. Recording action verification and analysis processes, accumulating all issues
    3. Saving memory to JSON files
    4. Supporting vector-based similarity search for similar scenarios
    """
    
    def __init__(self, memory_file="worldmodel_memory.json", embedding_cache_file="embedding_cache.pkl"):
        """
        Initialize the World Model Memory Manager
        
        Args:
            memory_file: Path to the memory storage file
            embedding_cache_file: Path to the embedding cache file
        """
        self.memory_file = memory_file
        self.embedding_cache_file = embedding_cache_file
        self.task_worldmodels = {}  # task_id -> world_model_summary
        self.task_accumulated_issues = {}  # task_id -> accumulated_issues
        self.action_analyses = {}  # task_id -> [action_analyses]
        self.scenario_embeddings = {}  # scenario_id -> embedding_vector
        
        # Try to load existing memory
        self._load_memory()
        self._load_embeddings()
    
    def _load_memory(self):
        """尝试从文件加载记忆"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                
                self.task_worldmodels = memory_data.get("task_worldmodels", {})
                self.task_accumulated_issues = memory_data.get("task_accumulated_issues", {})
                self.action_analyses = memory_data.get("action_analyses", {})
                print(f"Successfully loaded memory file: {self.memory_file}")
            except Exception as e:
                print(f"Failed to load memory file: {e}")
                # Initialize empty memory
                self.task_worldmodels = {}
                self.task_accumulated_issues = {}
                self.action_analyses = {}
    
    def _load_embeddings(self):
        """尝试从文件加载向量嵌入缓存"""
        if os.path.exists(self.embedding_cache_file):
            try:
                with open(self.embedding_cache_file, 'rb') as f:
                    self.scenario_embeddings = pickle.load(f)
                print(f"Successfully loaded {len(self.scenario_embeddings)} embeddings from cache")
            except Exception as e:
                print(f"Failed to load embeddings cache: {e}")
                self.scenario_embeddings = {}
    
    def save_memory(self):
        """Save memory to file"""
        memory_data = {
            "task_worldmodels": self.task_worldmodels,
            "task_accumulated_issues": self.task_accumulated_issues,
            "action_analyses": self.action_analyses,
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            
            # 保存嵌入向量缓存
            with open(self.embedding_cache_file, 'wb') as f:
                pickle.dump(self.scenario_embeddings, f)
                
            print(f"Successfully saved memory to file: {self.memory_file}")
            return True
        except Exception as e:
            print(f"Failed to save memory file: {e}")
            return False
    
    def save_worldmodel(self, task_id, worldmodel_summary):
        """
        Save mapping between task ID and world model summary
        
        Args:
            task_id: Task ID
            worldmodel_summary: World model summary text
        """
        task_id_str = str(task_id)  # Ensure key is string
        self.task_worldmodels[task_id_str] = worldmodel_summary
        
        # 同时计算并保存嵌入向量
        self._compute_and_store_embedding(task_id_str, worldmodel_summary)
        
        # Initialize task accumulated issues (if not exists)
        if task_id_str not in self.task_accumulated_issues:
            self.task_accumulated_issues[task_id_str] = {
                "object_existence": [],  # Objects don't exist or can't be found
                "object_state": [],      # Objects in wrong states
                "action_sequence": [],   # Actions in wrong order
                "spatial_relationship": [], # Missing required spatial relationships
                "object_accessibility": [], # Objects can't be accessed
                "precondition_violation": [], # Action preconditions not satisfied
                "object_property": [],   # Objects with wrong properties
                "other": []              # Other miscellaneous issues
            }
            
        self.save_memory()
        return task_id_str
    
    def _compute_and_store_embedding(self, task_id, text):
        """
        计算文本的嵌入向量并存储
        
        Args:
            task_id: 任务ID
            text: 要嵌入的文本
        """
        try:
            embedding = self._get_embedding(text)
            self.scenario_embeddings[task_id] = embedding
            return True
        except Exception as e:
            print(f"Error computing embedding for task {task_id}: {e}")
            return False
    
    def _get_embedding(self, text):
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            嵌入向量
        """
        # 使用OpenAI API获取文本嵌入
        try:
            # 如果可以使用OpenAI API
            response = openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002"
            )
            embedding = response['data'][0]['embedding']
            return embedding
        except Exception as e:
            # 降级方案：如果无法访问OpenAI API，使用简单的词频向量化
            print(f"Failed to get OpenAI embedding, using fallback method: {e}")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(max_features=1536)  # 匹配OpenAI维度
            vector = vectorizer.fit_transform([text]).toarray()[0]
            # 填充至1536维度
            if len(vector) < 1536:
                vector = np.pad(vector, (0, 1536 - len(vector)))
            return vector.tolist()
    
    def get_worldmodel(self, task_id):
        """
        Get world model summary for specified task ID
        
        Args:
            task_id: Task ID
        
        Returns:
            Corresponding world model summary, or None if not found
        """
        task_id_str = str(task_id)
        return self.task_worldmodels.get(task_id_str)
    
    def retrieve_similar_worldmodels(self, query_text, top_k=3, similarity_threshold=0.7):
        """
        基于文本相似度检索相似的世界模型
        
        Args:
            query_text: 查询文本
            top_k: 返回的最相似结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            包含任务ID、世界模型描述和相似度分数的列表
        """
        if not self.scenario_embeddings:
            print("No embeddings available for similarity search")
            return []
            
        # 计算查询文本的嵌入
        query_embedding = self._get_embedding(query_text)
        
        # 计算相似度
        similarities = []
        for task_id, embedding in self.scenario_embeddings.items():
            if embedding and len(embedding) > 0:
                # 确保维度匹配
                min_len = min(len(query_embedding), len(embedding))
                sim_score = cosine_similarity(
                    [query_embedding[:min_len]], 
                    [embedding[:min_len]]
                )[0][0]
                
                if sim_score >= similarity_threshold:
                    similarities.append((task_id, sim_score))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k结果
        results = []
        for task_id, score in similarities[:top_k]:
            worldmodel = self.get_worldmodel(task_id)
            if worldmodel:
                results.append({
                    "task_id": task_id,
                    "worldmodel": worldmodel,
                    "similarity": score,
                    "issues": self.get_accumulated_issues(task_id)
                })
        
        return results
    
    def get_enhanced_worldmodel(self, task_id):
        """
        Get enhanced world model summary including accumulated issues
        
        Args:
            task_id: Task ID
        
        Returns:
            Enhanced world model text
        """
        task_id_str = str(task_id)
        base_worldmodel = self.get_worldmodel(task_id_str)
        
        if not base_worldmodel:
            return None
        
        # Get accumulated issues
        accumulated_issues = self.task_accumulated_issues.get(task_id_str, {})
        
        # Build enhanced world model
        enhanced_worldmodel = base_worldmodel
        
        # Add accumulated issues and notes
        if any(issues for issues in accumulated_issues.values()):
            issues_summary = "\n\nACCUMULATED TASK ISSUES AND NOTES:\n"
            
            # Map category names to more readable titles
            category_titles = {
                "object_existence": "OBJECT EXISTENCE ISSUES",
                "object_state": "OBJECT STATE ISSUES",
                "action_sequence": "ACTION SEQUENCE ISSUES",
                "spatial_relationship": "SPATIAL RELATIONSHIP ISSUES",
                "object_accessibility": "OBJECT ACCESSIBILITY ISSUES",
                "precondition_violation": "PRECONDITION VIOLATION ISSUES",
                "object_property": "OBJECT PROPERTY ISSUES",
                "other": "OTHER ISSUES"
            }
            
            for issue_type, issues in accumulated_issues.items():
                if issues:
                    issues_summary += f"\n{category_titles.get(issue_type, issue_type.upper())}:\n"
                    for i, issue in enumerate(issues, 1):
                        issues_summary += f"{i}. {issue}\n"
            
            enhanced_worldmodel += issues_summary
        
        return enhanced_worldmodel
    
    def categorize_issue(self, issue_text):
        """
        Categorize an issue based on its text content
        
        Args:
            issue_text: Issue description text
            
        Returns:
            Category name
        """
        issue_text = issue_text.lower()
        
        # Define keywords for each category
        categories = {
            "object_existence": ["not exist", "doesn't exist", "don't exist", "cannot find", "not found", "no such", "missing object"],
            "object_state": ["wrong state", "incorrect state", "already open", "already closed", "already on", "already off", "should be open", "should be closed"],
            "action_sequence": ["sequence", "order", "before", "after", "first", "then", "prior", "preceding", "following", "preceded by"],
            "spatial_relationship": ["close", "proximity", "facing", "adjacent", "nearby", "relation", "relationship", "position"],
            "object_accessibility": ["cannot reach", "can't reach", "inaccessible", "not accessible", "access"],
            "precondition_violation": ["precondition", "requirement", "required", "necessary condition", "must be done"],
            "object_property": ["property", "type", "not grabable", "not a container", "cannot be opened", "cannot be closed"]
        }
        
        # Check for category keywords
        for category, keywords in categories.items():
            if any(keyword in issue_text for keyword in keywords):
                return category
        
        # Default category
        return "other"
    
    def save_action_analysis(self, task_id, action_plan, verification_response):
        """
        Save action plan analysis results and accumulate issues
        
        Args:
            task_id: Task ID
            action_plan: Action plan text
            verification_response: Verification response text
            
        Returns:
            Analysis result dictionary
        """
        task_id_str = str(task_id)
        
        # Extract structured verification result
        analysis = extract_verification_result(verification_response)
        
        # Add metadata
        analysis["task_id"] = task_id_str
        analysis["action_plan"] = action_plan
        analysis["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        # 保存行动计划的嵌入向量，用于未来相似性搜索
        combined_text = f"{action_plan}\n{verification_response}"
        self._compute_and_store_embedding(f"{task_id_str}_action", combined_text)
        
        # Save to memory
        if task_id_str not in self.action_analyses:
            self.action_analyses[task_id_str] = []
        
        self.action_analyses[task_id_str].append(analysis)
        
        # Initialize accumulated issues (if not exists)
        if task_id_str not in self.task_accumulated_issues:
            self.save_worldmodel(task_id, "")  # This will initialize the accumulated issues structure
        
        # Process each issue and categorize it
        all_issues = []
        
        # Process structured issues
        for issue_type, issues in analysis["issues"].items():
            for issue in issues:
                if issue and len(issue.strip()) > 0:
                    all_issues.append(issue)
        
        # Process environment change as issue if verification failed
        if not analysis["is_executable"] and analysis["environment_change"] and len(analysis["environment_change"].strip()) > 0:
            if not any(analysis["environment_change"] in issue for issue in all_issues):
                all_issues.append(analysis["environment_change"])
        
        # Categorize and accumulate each issue
        for issue in all_issues:
            category = self.categorize_issue(issue)
            
            # Add issue if not already present
            if issue not in self.task_accumulated_issues[task_id_str][category]:
                self.task_accumulated_issues[task_id_str][category].append(issue)
        
        self.save_memory()
        
        return analysis
    
    def retrieve_similar_action_analyses(self, action_plan, top_k=3, similarity_threshold=0.7):
        """
        基于相似度检索类似的行动计划分析
        
        Args:
            action_plan: 当前行动计划
            top_k: 返回的最相似结果数量
            similarity_threshold: 相似度阈值
            
        Returns:
            包含任务ID、行动计划、分析结果和相似度分数的列表
        """
        # 遍历所有嵌入向量，寻找行动计划分析
        action_embeddings = {k: v for k, v in self.scenario_embeddings.items() if "_action" in k}
        
        if not action_embeddings:
            print("No action analysis embeddings available")
            return []
            
        # 计算查询文本的嵌入
        query_embedding = self._get_embedding(action_plan)
        
        # 计算相似度
        similarities = []
        for embed_id, embedding in action_embeddings.items():
            if embedding and len(embedding) > 0:
                # 确保维度匹配
                min_len = min(len(query_embedding), len(embedding))
                sim_score = cosine_similarity(
                    [query_embedding[:min_len]], 
                    [embedding[:min_len]]
                )[0][0]
                
                if sim_score >= similarity_threshold:
                    task_id = embed_id.split("_action")[0]
                    similarities.append((task_id, sim_score))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k结果
        results = []
        for task_id, score in similarities[:top_k]:
            # 找到最新的分析结果
            analyses = self.get_action_analyses(task_id)
            if analyses:
                latest_analysis = analyses[-1]
                results.append({
                    "task_id": task_id,
                    "action_plan": latest_analysis.get("action_plan", ""),
                    "analysis": latest_analysis,
                    "similarity": score
                })
        
        return results
    
    def get_action_analyses(self, task_id):
        """Get all action analyses for specified task ID"""
        task_id_str = str(task_id)
        return self.action_analyses.get(task_id_str, [])
    
    def get_latest_analysis(self, task_id):
        """Get the latest action analysis for specified task ID"""
        analyses = self.get_action_analyses(task_id)
        if analyses:
            return analyses[-1]
        return None
    
    def get_accumulated_issues(self, task_id):
        """Get accumulated issues for specified task ID"""
        task_id_str = str(task_id)
        return self.task_accumulated_issues.get(task_id_str, {})
    
    def reset_task_issues(self, task_id):
        """Reset accumulated issues for specified task ID"""
        task_id_str = str(task_id)
        if task_id_str in self.task_accumulated_issues:
            self.task_accumulated_issues[task_id_str] = {
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


def create_verification_prompt(world_model_str, action_plan_str, similar_scenarios=None):
    """
    Create structured verification prompt with similar scenarios
    
    Args:
        world_model_str: World model text representation
        action_plan_str: Action plan to verify
        similar_scenarios: List of similar scenarios with verification results
        
    Returns:
        Structured verification prompt
    """
    prompt = """
Please carefully evaluate the following action plan based on the provided world model. 
For each action in the plan, verify that all necessary conditions for successful execution are met, 
considering the following rules:

1. **Existence and State Checks:**
   - Each target object mentioned in the action must exist in the world model.
   - The object's current state must be appropriate for the intended action. 
   For example:
     - For a [grab] action, the object must be grabbable, not already held, 
     and there must be an associated "CLOSE" relationship between the agent and the object.
     - For an [open] action, the object must be a container (from objects_inside) and currently in a CLOSED state.
     - For a [close] action, the object must be open.
     - For a [switchon] action, the object must be off.
     - For [putin] or [putback] actions, ensure that the agent has previously grabbed the object, is in the correct proximity to the container (verified by a "CLOSE" edge), and that the container is in the appropriate state (e.g., open for putin, or of the correct type for putback).

2. **Sequential and Logical Dependencies:**
   - Verify that the sequence of actions follows logical dependencies. 
    For example:
     - A [grab] action should be preceded by a [walk] (or equivalent movement) to the target object.
     - A [putin] or [putback] action must follow a successful [grab] action.
     - If an action such as [switchon] is executed, the object must not already be in the target state.
   - Check that the actions are arranged in an order that reflects the necessary preconditions and postconditions for each action.

3. **Spatial Relationships and Consistency:**
   - Ensure that the spatial relationships described in the world model (e.g., INSIDE, CLOSE, FACING) are consistent with the requirements of each action.
   - For instance, if an action requires the agent to be close to an object (e.g., for grabbing or interacting), verify that the world model indicates an appropriate CLOSE relationship.
"""

    # 添加相似场景的经验
    if similar_scenarios and len(similar_scenarios) > 0:
        prompt += """
4. **Experience from Similar Scenarios:**
   Consider the following verification results from similar scenarios that have been analyzed previously:
"""
        for i, scenario in enumerate(similar_scenarios, 1):
            prompt += f"""
   Scenario {i}:
   - Similar World Model: {scenario.get('worldmodel', '')}
   - Verification Result: {'Executable' if scenario.get('analysis', {}).get('is_executable', False) else 'Not Executable'}
   - Key Issues: {'; '.join([issue for issues in scenario.get('issues', {}).values() for issue in issues[:2]])}
"""

    prompt += """
Please provide your evaluation in the following structured format:

"The answer is <yes or no>.
Environment change: <detailed description of any predicted changes in the environment or specific reasons why the plan is not executable>

Issues:
- EXISTENCE ERROR: <describe any objects that don't exist or can't be found>
- STATE ERROR: <describe any objects in wrong states>
- SEQUENCE ERROR: <describe any actions in wrong order>
- RELATIONSHIP ERROR: <describe any missing required spatial relationships>
- ACCESSIBILITY ERROR: <describe any objects that can't be accessed>
- PRECONDITION ERROR: <describe any unsatisfied action preconditions>
- PROPERTY ERROR: <describe any objects with wrong properties for the intended actions>
"

If there are no issues of a particular type, omit that line. If the action sequence is empty, please answer "The answer is no" with appropriate explanation.

There may be some "def task()" in the action sequence, which is reasonable because the algorithm splits the whole goal into separate tasks.
Don't be too strict in your evaluation.

World Model:
{world_model}

Action Sequence to Evaluate:
{action_plan}
"""
    
    # Fill prompt template
    filled_prompt = prompt.format(
        world_model=world_model_str,
        action_plan=action_plan_str
    )
    
    return filled_prompt
def verify_action_plan(world_model_str, action_plan_str, memory=None, max_retries=3, model="gpt-35-turbo"):
    """
    Verify if an action plan is executable in the given world model
    
    Args:
        world_model_str: World model text representation
        action_plan_str: Action plan to verify
        memory: WorldModelMemory instance
        max_retries: Maximum number of retries
        model: LLM model name
        
    Returns:
        Tuple of (verification_result, is_executable)
    """
    # 检索相似场景（如果提供了memory实例）
    similar_scenarios = []
    if memory:
        try:
            # 首先尝试检索相似的行动计划分析
            similar_action_analyses = memory.retrieve_similar_action_analyses(action_plan_str, top_k=2)
            
            # 然后检索相似的世界模型
            similar_worldmodels = memory.retrieve_similar_worldmodels(world_model_str, top_k=2)
            
            # 合并结果，去重
            task_ids_seen = set()
            for item in similar_action_analyses + similar_worldmodels:
                task_id = item.get("task_id")
                if task_id and task_id not in task_ids_seen:
                    task_ids_seen.add(task_id)
                    similar_scenarios.append(item)
                    
            print(f"Found {len(similar_scenarios)} similar scenarios in memory")
        except Exception as e:
            print(f"Error retrieving similar scenarios: {e}")
    
    # Build structured verification prompt with similar scenarios
    prompt = create_verification_prompt(world_model_str, action_plan_str, similar_scenarios)
    
    # Call LLM API to get verification result
    retries = 0
    while retries < max_retries:
        try:
            # Use get_response from test_openai
            from test_openai import get_response
            raw_response = get_response(prompt)
            verification_result = raw_response.strip()
            
            # Debug output
            print("Generated Prompt:\n", prompt)
            print("LLM Response:\n", verification_result)
            
            # 解析验证结果，确定是否可执行
            is_executable = "The answer is yes" in verification_result.lower()
            
            # 无论验证结果如何，都保存到记忆（如果提供了memory实例）
            if memory:
                # 生成唯一的任务ID
                task_id = f"task_{int(time.time())}"
                # 保存世界模型
                memory.save_worldmodel(task_id, world_model_str)
                # 保存行动分析
                memory.save_action_analysis(task_id, action_plan_str, verification_result)
                
                # 如果不可执行，记录详细的失败原因和环境状态
                if not is_executable:
                    analysis = extract_verification_result(verification_result)
                    failure_reasons = [issue for issues in analysis["issues"].values() for issue in issues]
                    print(f"Plan verification failed. Reasons: {', '.join(failure_reasons)}")
            
            # 返回验证结果和可执行状态
            return verification_result, is_executable
            
        except Exception as e:
            retries += 1
            time.sleep(1)
            print(f"LLM API call failed (attempt {retries}/{max_retries}): {e}")
    
    # Return default negative response when all retries failed
    default_response = "The answer is no.\nEnvironment change: Failed to get response from LLM after multiple retries."
    print("Warning: All LLM API calls failed, returning default response")
    return default_response, False

def extract_verification_result(verification_response):
    """
    Extract structured result from LLM verification response
    
    Args:
        verification_response: LLM verification response text
        
    Returns:
        Dictionary containing verification result
    """
    result = {
        "is_executable": False,
        "environment_change": "",
        "issues": {
            "existence": [],
            "state": [],
            "sequence": [],
            "relationship": [],
            "accessibility": [],
            "precondition": [],
            "property": [],
            "other": []
        },
        "raw_response": verification_response
    }
    
    # Extract yes/no answer
    answer_match = re.search(r"The answer is (yes|no)", verification_response, re.IGNORECASE)
    if answer_match:
        result["is_executable"] = answer_match.group(1).lower() == "yes"
    
    # Extract environment change/execution issue description
    env_change_match = re.search(r"Environment change: (.*?)($|\n\n|\nIssues:)", verification_response, re.DOTALL)
    if env_change_match:
        result["environment_change"] = env_change_match.group(1).strip()
    
    # Extract various issue types
    issue_patterns = {
        "existence": r"EXISTENCE ERROR: (.*?)($|\n-|\n\n)",
        "state": r"STATE ERROR: (.*?)($|\n-|\n\n)",
        "sequence": r"SEQUENCE ERROR: (.*?)($|\n-|\n\n)",
        "relationship": r"RELATIONSHIP ERROR: (.*?)($|\n-|\n\n)",
        "accessibility": r"ACCESSIBILITY ERROR: (.*?)($|\n-|\n\n)",
        "precondition": r"PRECONDITION ERROR: (.*?)($|\n-|\n\n)",
        "property": r"PROPERTY ERROR: (.*?)($|\n-|\n\n)"
    }
    
    for issue_type, pattern in issue_patterns.items():
        match = re.search(pattern, verification_response, re.DOTALL)
        if match:
            result["issues"][issue_type].append(match.group(1).strip())
    
    # If no specific issues found but verification failed,
    # extract general issues from environment change
    if not result["is_executable"] and result["environment_change"] and not any(result["issues"].values()):
        result["issues"]["other"].append(result["environment_change"])
    
    return result


def build_enhanced_worldmodel(args, vh_envs, sample, memory, logging=None):
    """
    Build enhanced World Model using LLMPolicy methods, integrated with memory system
    
    Args:
        args: Program arguments
        vh_envs: Virtual environment object
        sample: Task sample ID
        memory: WorldModelMemory instance
        logging: Logging object (optional)
        
    Returns:
        merged: Enhanced World Model dictionary
        merged_str: World Model text representation
    """
    from copy import deepcopy
    from llm_policy import LLMPolicy, split_goal
    from data_ti import remove_duplicates, sort_graph
    
    task_id_str = str(sample)
    
    # First check if world model exists in memory
    existing_worldmodel = memory.get_worldmodel(sample)
    if existing_worldmodel:
        print(f"Loading world model for task {sample} from memory")
        # Get enhanced world model (including accumulated issues)
        enhanced_worldmodel = memory.get_enhanced_worldmodel(sample)
        # Return empty merged dict and enhanced text
        return {}, enhanced_worldmodel
    
    # 检查是否有相似场景缓存
    similar_worldmodels = memory.retrieve_similar_worldmodels(str(sample), top_k=1, similarity_threshold=0.9)
    if similar_worldmodels:
        print(f"Found very similar world model for task {sample}")
        similar_model = similar_worldmodels[0]
        # 使用相似场景的世界模型
        enhanced_worldmodel = similar_model["worldmodel"]
        # 添加相似场景的问题提示
        if similar_model["issues"]:
            enhanced_worldmodel += "\n\nNOTE: Similar scenario had following issues:\n"
            for issue_type, issues in similar_model["issues"].items():
                if issues:
                    enhanced_worldmodel += f"- {issue_type.upper()}: {issues[0]}\n"
        return {}, enhanced_worldmodel
    
    # Load environment using task ID
    obs, env_graph = vh_envs.reset(task_id=sample)
    
    # Filter observation nodes
    if 'nodes' in obs[0]:
        from utils_bc.utils_graph import filter_redundant_nodes
        obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
    
    # Create LLMPolicy instance for processing
    llm_policy = LLMPolicy(args, logging)
    llm_policy.reset(ids=sample)
    llm_policy.set_graph(env_graph)
    llm_policy.set_goal(vh_envs.task_goal[0])
    
    # Decompose task goal for more detailed information
    split_goals, _ = split_goal(logging, llm_policy.task_goal)
    
    # Collect all relevant object information
    for goal in split_goals:
        llm_policy.get_goal_obj_message(goal)
    
    # Build enhanced World Model based on original environment graph and LLMPolicy extracted information
    merged = {
        "nodes": deepcopy(env_graph["nodes"]),
        "edges": deepcopy(env_graph["edges"])
    }
    
    # Remove duplicates and sort
    remove_duplicates(merged)
    sort_graph(merged)
    
    # Generate basic World Model text representation
    merged_str = summarize_observation(merged)
    
    # Add supplementary object relations and states
    if llm_policy.goal_objs_loc:
        obj_relations = "\n\nAdditional Object Relationships:\n"
        for loc in llm_policy.goal_objs_loc:
            if len(loc) >= 3:
                obj_relations += f"- {loc[0]} {loc[1]} {loc[2]}\n"
        merged_str += obj_relations
    
    # Save world model to memory
    memory.save_worldmodel(sample, merged_str)
    
    # Get enhanced world model (at this point should only have basic info, no accumulated issues)
    enhanced_worldmodel = memory.get_enhanced_worldmodel(sample)
    
    return merged, enhanced_worldmodel


def summarize_observation(obs):
    """
    将简化后的观测数据转换为文本摘要，
    包括物体（节点）的基本信息和物体之间的关系（边）。
    """
    summary = "World Model Summary:\n"
    
    summary += "Objects:\n"
    for node in obs.get("nodes", []):
        states = ", ".join(node.get("states", [])) if node.get("states", []) else "None"
        summary += f"- ID {node['id']}: {node['class_name']}, States: {states}\n"
    
    summary += "\nRelationships:\n"
    for edge in obs.get("edges", []):
        summary += f"- {edge['from_id']} {edge['relation_type']} {edge['to_id']}\n"
    
    return summary