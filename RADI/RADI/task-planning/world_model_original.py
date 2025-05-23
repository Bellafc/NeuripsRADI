
import os
import time
import openai

# def verify_action_plan(world_model: str, action_list: str, max_retries=3, model="gpt-3.5-turbo") -> bool:
def verify_action_plan(world_model: str, action_list: str, max_retries=3, model="gpt-35-turbo") -> bool:
    # 1. 将动作计划按行拆分，并过滤掉空行和注释行
    actions = [line.strip() for line in action_list.splitlines()
               if line.strip() and not line.strip().startswith('#')]
    
    # 2. 生成验证 prompt
    prompt = f"""
Please carefully evaluate the following action plan based on the provided world model. 
For each action in the plan, verify that all necessary conditions for successful execution are met, 
Considering the following rules:

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

4. **Overall Plan Evaluation:**
   - After examining all individual actions and their dependencies, determine whether the entire plan is logically consistent and executable.
   - If any condition is not met (e.g., a required object is missing, the state is incorrect, or the sequence is illogical), the plan should be considered unexecutable.

Please provide your evaluation in the following strict format:

"The answer is <yes or no>.
Environment change: <detailed description of any predicted changes in the environment or specific reasons why the plan is not executable>"

If the action sequence is empty, please answer "THe answer is no".


World Model:
{world_model}

This is the action sequence to be evaluated:
"""


    prompt = f"""
Please carefully evaluate the following action plan based on the provided world model. 
For each action in the plan, verify that all necessary conditions for successful execution are met, 
Considering the following rules:

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

Please provide your evaluation in the following strict format:

"The answer is <yes or no>.
Environment change: <detailed description of any predicted changes in the environment or specific reasons why the plan is not executable>"

If the action sequence is empty, please answer "The answer is no".

There're some "def task()" in the action sequence, which is reasonable, because the algorithm split the whole goal into few tasks.
Don't be too strict.
World Model:
{world_model}

This is the action sequence to be evaluated:
"""
    for idx, action in enumerate(actions, start=1):
        prompt += f"{idx}. {action}\n"
    
    # 3. 调用 OpenAI ChatCompletion API 获取响应
    retries = 0
    while retries < max_retries:
        try:
            # response = openai.ChatCompletion.create(
            #     model=model,
            #     messages=[{"role": "user", "content": prompt}],
            #     temperature=0.0
            # )
            # raw_response = response["choices"][0]["message"]["content"]
            # from test_openai import get_response
            # raw_response = get_response(prompt, model=model, temperature=0)
            from test_openai import get_response
            raw_response = get_response(prompt)
            verification_result = raw_response.strip()
            
            # 调试输出（根据需要可注释掉）
            print("Generated Prompt:\n", prompt)
            print("LLM Response:\n", verification_result)
            
            # 4. 简单判断：如果响应中包含 "yes" 则认为计划可执行
            answer_lower = verification_result.lower()
            return "no" not in answer_lower
        except Exception as e:
            retries += 1
            time.sleep(1)
    
    raise Exception("LLM API call failed after retries.")
