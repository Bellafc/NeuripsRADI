import copy
import glob
import os, sys
import time
import numpy as np
import torch
import torch.nn.functional as F

import pdb
import pickle
import json
import random
from copy import deepcopy

from utils_bc import utils_interactive_eval
from utils_bc.utils_graph import filter_redundant_nodes
from envs.utils.check_logical import check_env_bug
from llm_policy import LLMPolicy, split_goal
from sim_compute import Similarity
from data_ti import remove_duplicates,sort_graph
from world_model import WorldModelMemory, verify_action_plan, generate_corrected_plan

worldmodel_obs = []
flag = 0
temp2 =[]
results = []
final_result_ob = ""
final_result_rs = ""
worldmodel_obs_ob =[]
worldmodel_obs_rs =[]
results_ob=[]
results_rs=[]
merged = {'nodes':[],'edges':[]}
def summarize_observation(obs):
    summary = "World Model Summary:\n"
    
    summary += "Objects:\n"
    for node in obs.get("nodes", []):
        states = ", ".join(node.get("states", [])) if node.get("states", []) else "None"
        summary += f"- ID {node['id']}: {node['class_name']}, States: {states}\n"
    
    summary += "\nRelationships:\n"
    for edge in obs.get("edges", []):
        summary += f"- {edge['from_id']} {edge['relation_type']} {edge['to_id']}\n"
    
    return summary
def aggregate_worldmodel_obs(obs_list):
    """
    将多个观测帧合并，返回唯一的节点和边，
    每个节点只保留 id、class_name 和 states，
    每条边只保留 from_id、relation_type 和 to_id。
    
    节点的状态会被合并，确保同一个节点的所有观测状态都被记录。
    """
    aggregated_nodes = {}
    aggregated_edges = set()
    
    for item in obs_list:
        for key, obs in item.items():
            for node in obs.get("nodes", []):
                node_id = node["id"]
                if node_id not in aggregated_nodes:
                    # 节点首次出现，初始化
                    aggregated_nodes[node_id] = {
                        "class_name": node.get("class_name"),
                        "states": set(node.get("states", []))
                    }
                else:
                    # 节点已存在，合并状态信息
                    aggregated_nodes[node_id]["states"].update(node.get("states", []))
            
            for edge in obs.get("edges", []):
                edge_key = (edge.get("from_id"), edge.get("relation_type"), edge.get("to_id"))
                aggregated_edges.add(edge_key)
    
    # 将节点状态从集合转换回列表
    nodes_list = [{"id": nid, "class_name": info["class_name"], "states": list(info["states"])}
                  for nid, info in aggregated_nodes.items()]
    edges_list = [{"from_id": f, "relation_type": rel, "to_id": t}
                  for (f, rel, t) in aggregated_edges]

    return [{"nodes": nodes_list, "edges": edges_list}]

    """
    将简化后的观测数据转换为文本摘要，
    包括物体（节点）的基本信息和物体之间的关系（边）。
    """
    summary = ""
    
    for edge in obs.get("edges", []):
        summary += f"{edge['from_id']} {edge['relation_type']} {edge['to_id']}; "
    return summary
def merge_observations(obs_list):
    """
   
    参数:
        obs_list: list，每个元素为 {'nodes': [...], 'edges': [...]} 或者嵌套的列表
    返回:
        合并后的 world model，以列表形式返回，例如：[{'nodes': [...], 'edges': [...]}]
    """
    world_model = {'nodes': [], 'edges': []}
    existing_node_ids = set()
    existing_edge_keys = set()

    def process_obs(obs):
        # 合并节点
        for node in obs.get('nodes', []):
            if node['id'] not in existing_node_ids:
                world_model['nodes'].append(node)
                existing_node_ids.add(node['id'])
        # 合并边
        for edge in obs.get('edges', []):
            key = (edge['from_id'], edge['relation_type'], edge['to_id'])
            if key not in existing_edge_keys:
                world_model['edges'].append(edge)
                existing_edge_keys.add(key)

    def traverse(item):
        """
        递归遍历数据：
        - 如果是字典且包含 "nodes" 和 "edges" 键，则处理；
        - 如果是列表，则逐个递归处理。
        """
        if isinstance(item, dict):
            if 'nodes' in item and 'edges' in item:
                process_obs(item)
            else:
                print('warning')
                
        elif isinstance(item, list):
            for sub_item in item:
                traverse(sub_item)
        else:
            print("warning", item)

    traverse(obs_list)
    return world_model
def compute_task_complexity(task_goal, graph):
    min_steps = 0
    for goal in task_goal:
        goal_num = task_goal[goal]
        if 'turn' in goal:
            min_steps += 1
        elif 'inside' in goal:
            obj_name = goal.split('_')[1]
            obj_num = goal_num
            # indide object num
            inside_num = 0
            # outside object num
            out_num = 0
            # judge object location
            for node in graph['nodes']:
                if node['class_name'] == obj_name:
                    obj_id = node['id']
                    from_obj_edges = [edge for edge in graph['edges'] if edge['from_id'] == obj_id]
                    for edge in reversed(from_obj_edges):
                        if edge['relation_type'] == 'INSIDE':
                            inside_num += 1
                            break
                        elif edge['relation_type'] == 'ON':
                            out_num += 1
                            break
            # use object outside first, due to its fewer action step 
            # obj inside: walk, open, grab, close, walk, open, putin, close
            # obj outside: walk, grab, walk, open, putin, close
            if obj_num <= out_num:
                min_steps += 6 * goal_num
            else:
                min_steps += 6 * out_num + 8 * (obj_num - out_num)
        elif 'on' in goal:
            obj_name = goal.split('_')[1]
            obj_num = goal_num
            inside_num = 0
            out_num = 0
            # pan duan obj wei zhi
            for node in graph['nodes']:
                if node['class_name'] == obj_name:
                    obj_id = node['id']
                    from_obj_edges = [edge for edge in graph['edges'] if edge['from_id'] == obj_id]
                    for edge in reversed(from_obj_edges):
                        if edge['relation_type'] == 'INSIDE':
                            inside_num += 1
                            break
                        elif edge['relation_type'] == 'ON':
                            out_num += 1
                            break
            # use object outside first, due to its fewer action step 
            # obj inside: walk, open, grab, close, walk, putback
            # obj outside: walk, grab, walk, putback
            if obj_num <= out_num:
                min_steps += 4 * obj_num
            else:
                min_steps += 4 * out_num + 6 * (obj_num - out_num)
    return min_steps
# def build_enhanced_worldmodel(args, vh_envs, sample, logging=None):
#     """
#     使用LLMPolicy中的方法构建增强的WorldModel
    
#     参数:
#         args: 程序参数
#         vh_envs: 虚拟环境对象
#         sample: 任务样本ID
#         logging: 日志对象(可选)
    
#     返回:
#         merged: 增强的WorldModel字典
#         merged_str: WorldModel的文本表示
#     """
#     from copy import deepcopy
#     from llm_policy import LLMPolicy, split_goal
    
#     # 使用任务ID加载环境
#     obs, env_graph = vh_envs.reset(task_id=sample)
    
#     # 对观察节点进行过滤
#     if 'nodes' in obs[0]:
#         obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
    
#     # 创建临时的LLMPolicy实例进行处理
#     llm_policy = LLMPolicy(args, logging)
#     llm_policy.reset(ids=sample)
#     llm_policy.set_graph(env_graph)
#     llm_policy.set_goal(vh_envs.task_goal[0])
    
#     # 通过LLMPolicy提取任务相关信息
#     # set_goal方法已填充了task_obj和goal_objs_loc
    
#     # 分解任务目标获取更详细信息
#     split_goals, _ = split_goal(logging, llm_policy.task_goal)
    
#     # 收集所有相关对象信息
#     all_objects = []
#     all_locations = []
#     all_states = []
    
#     for goal in split_goals:
#         obj_loc, obj_state, obj_list = llm_policy.get_goal_obj_message(goal)
#         all_objects.extend(obj_list)
        
#         # 将位置和状态描述添加到列表中(如果非空)
#         if obj_loc:
#             all_locations.append(obj_loc)
#         if obj_state:
#             all_states.append(obj_state)
    
#     # 基于原始环境图和LLMPolicy提取的信息构建增强的WorldModel
#     merged = {
#         "nodes": deepcopy(env_graph["nodes"]),
#         "edges": deepcopy(env_graph["edges"])
#     }
    
#     # 去除重复项和排序
#     remove_duplicates(merged)
#     sort_graph(merged)
    
#     # 生成基本的WorldModel文本表示
#     merged_str = summarize_observation(merged)
    
#     # 添加对象关系和状态的补充描述
#     if llm_policy.goal_objs_loc:
#         obj_relations = "\n\nAdditional Object Relationships:\n"
#         for loc in llm_policy.goal_objs_loc:
#             if len(loc) >= 3:
#                 obj_relations += f"- {loc[0]} {loc[1]} {loc[2]}\n"
#         merged_str += obj_relations
    
#     # 添加对象状态补充描述
#     if all_locations or all_states:
#         obj_details = "\n\nAdditional Object Details:\n"
        
#         if all_locations:
#             obj_details += "Locations:\n"
#             for loc in all_locations:
#                 obj_details += f"- {loc}\n"
        
#         if all_states:
#             obj_details += "States:\n"
#             for state in all_states:
#                 obj_details += f"- {state}\n"
                
#         merged_str += obj_details
    
#     return merged, merged_str
def build_enhanced_worldmodel(args, vh_envs, sample, logging=None):
    """
    使用LLMPolicy构建精简的、只包含任务相关信息的WorldModel

    参数:
        args: 程序参数
        vh_envs: 虚拟环境对象
        sample: 任务样本ID
        logging: 日志对象(可选)

    返回:
        worldmodel: 包含任务相关对象、位置、状态、关系的字典
        worldmodel_str: WorldModel 的文本表示
    """
    from llm_policy import LLMPolicy, split_goal

    # 重置环境并获取初始观察和环境图
    obs, env_graph = vh_envs.reset(task_id=sample)

    # 如果有多余节点，可以先过滤（可选）
    if 'nodes' in obs[0]:
        obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])

    # 初始化 LLMPolicy
    llm = LLMPolicy(args, logging)
    llm.reset(ids=sample)
    llm.set_graph(env_graph)
    llm.set_goal(vh_envs.task_goal[0])

    # 拆分子目标，聚焦细节
    split_goals, _ = split_goal(logging, llm.task_goal)

    # 收集任务相关的信息
    all_objects = []
    all_locations = []
    all_states = []
    obj_relations = []

    for goal in split_goals:
        loc_msg, state_msg, objs = llm.get_goal_obj_message(goal)
        all_objects.extend(objs)

        if loc_msg:
            all_locations.append(loc_msg)
            # 假设格式 loc_msg = (obj, relation, target)
            if isinstance(loc_msg, (list, tuple)) and len(loc_msg) >= 3:
                obj_relations.append(loc_msg)

        if state_msg:
            all_states.append(state_msg)

    # 去重
    all_objects = sorted(set(all_objects))
    all_locations = sorted(set(all_locations))
    all_states = sorted(set(all_states))
    obj_relations = sorted(set(obj_relations))

    # 构建精简的 worldmodel
    worldmodel = {
        "objects": all_objects,
        "locations": all_locations,
        "states": all_states,
        "relations": obj_relations
    }

    # 生成文本表示
    lines = ["WorldModel (任务相关部分):"]
    if all_objects:
        lines.append("Objects:")
        for o in all_objects:
            lines.append(f"- {o}")
    if obj_relations:
        lines.append("\nRelations:")
        for subj, rel, obj in obj_relations:
            lines.append(f"- {subj} {rel} {obj}")
    if all_locations:
        lines.append("\nLocations:")
        for loc in all_locations:
            lines.append(f"- {loc}")
    if all_states:
        lines.append("\nStates:")
        for st in all_states:
            lines.append(f"- {st}")

    worldmodel_str = "\n".join(lines)
    return worldmodel, worldmodel_str

def llm_evaluation(args, vh_envs, logging):
    global flag
    global worldmodel_obs
    global results_ob
    global results_rs
    global merged_str
    
    llm_policy = LLMPolicy(args,logging)
    # control flags
    if_exe_all_action = True
    verbose = True
    valid_run = 0
    success_count = 0
    camera_num = vh_envs.comm.camera_count()[1]
    
    # set test_examples
    testid=[]
    with open('FADI/FADI/task-planning/test/'+args.subset+'.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            testid.append(line.strip())

    '''
    testid=[]
    with open('debug/'+args.subset+'.txt','r',encoding='utf-8') as f:
        for line in f.readlines():
            testid.append(line.strip())
    '''
    # task number
    i = 0
    # iterate testid
    drop = 0
    for sample in testid:
        merged = {'nodes':[],'edges':[]}
        merged_str = ""
        sample=int(sample)
        print('*'*30,"New Sample","*"*30)
        verify_attempt = 0
        max_verify_attempt = 99  # 可根据需要调整
        valid_run_tem = False
        success_run_tem = False
        all_cur_observation = []
    # 初始调用reset获取第一个观察
        obs, env_graph = vh_envs.reset(task_id=sample)
        # obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
        # all_cur_observation.append(deepcopy(obs))
        # temp = aggregate_worldmodel_obs(all_cur_observation)
        # merged_world_model = merge_observations(temp)
        # merged = {"nodes": [], "edges": []}
        # merged["nodes"].extend(merged_world_model["nodes"])
        # merged["edges"].extend(merged_world_model["edges"])
        # remove_duplicates(merged)
        # sort_graph(merged)
        # # print("merged_world_model")
        # merged_str = summarize_observation(merged)
        # print(merged_str)
        from world_model import verify_action_plan
        from world_model import WorldModelMemory
        memory = WorldModelMemory(memory_file="worldmodel_memory.json")
        memory.clear_all_memory()
        merged, merged_str = build_enhanced_worldmodel(args, vh_envs, sample,logging)

        while verify_attempt < max_verify_attempt:
            steps = 0
            if verify_attempt > 0:
                
                # temp = aggregate_worldmodel_obs(all_cur_observation)
                # merged_world_model = merge_observations(temp)
                # merged = {"nodes": [], "edges": []}
                # merged["nodes"].extend(merged_world_model["nodes"])
                # merged["edges"].extend(merged_world_model["edges"])
                # remove_duplicates(merged)
                # sort_graph(merged)
                # # print("merged_world_model")
                # merged_str = summarize_observation(merged)
                # # print(merged_str)
                merged, merged_str = build_enhanced_worldmodel(args, vh_envs, sample,logging)

            valid_run_tem = False
            success_run_tem = False
    
             # -----------compute task complexity-------------------------
            task_goal = vh_envs.task_goal[0]
            graph = env_graph
            complexity = compute_task_complexity(task_goal, graph)
            # print('Current Task id: {}, Task Goal: {}, Min Step: {}'.format(sample,task_goal,complexity))
            # --------------
            # set gpt policy
            # --------------
            llm_policy.reset(ids=sample)
            llm_policy.set_graph(env_graph)  # 设置gpt任务环境
            llm_policy.set_goal(vh_envs.task_goal[0])  # 设置gpt任务目标
            
            if args.mode=='multi-layer':
                llm_policy.generate_multi_layer_plan()
            from llm_policy import de_final_complete_plan
  
            
            verification_response, is_executable = verify_action_plan(merged_str, de_final_complete_plan,memory=memory)
            print(f"DEBUG: Verification result is_executable = {is_executable}")
            # print('*'*20,'Execute Action List','*'*20)
            if is_executable: ####not???
                # 验证成功：清空merged，并进入动作执行循环
                memory.clear_memory_for_task(sample)
                merged = {"nodes": [], "edges": []}
                print('*' * 20, 'Execute Action List', '*' * 20)
                
                # -------------------- 动作执行循环 --------------------
                while True:
                    agent_id = 0
                    agent_actions = {}
                    agent_rewards = {}
                    agent_ignore_walk = {}
                    ignore_walk = None
                    action_obj_str = ''
        
                    if if_exe_all_action:
                        llm_action_obj_str = llm_policy.get_action_from_llm()
                        if llm_action_obj_str != 'DONE':
                            env_task_goal_write = ['%s_%d' % (k, v) for k, v in task_goal.items() if v > 0]
                            print('-' * 100)
                            
                            print('Step: {}, Task: {}'.format(steps + 1, str(env_task_goal_write)))
                            print('Action: {}'.format(llm_action_obj_str))
                        else:
                            valid_run_tem = True
                            break
                    else:
                        # 若非收集全部动作模式（此处可扩展其他逻辑）
                        llm_action_obj_str = llm_policy.get_action_from_llm()
        
                    action_obj_str = llm_action_obj_str
                    agent_actions[agent_id] = action_obj_str
                    agent_ignore_walk[agent_id] = ignore_walk
        
        
                    # 发送动作到环境
                    obs, rewards, dones, infos, success = vh_envs.step(
                        agent_actions,
                        ignore_walk=agent_ignore_walk,
                        logging=logging
                    )
        
                    if rewards == dones == infos == success == None:
                        print('Action Fail')
                        print('Fail Reason: {}'.format(json.dumps(obs)))
                        valid_run_tem = False
                        break
        
                    # 检查动作执行后的环境状态
                    obs[0]['nodes'] = filter_redundant_nodes(obs[0]['nodes'])
                    env_bug_count_a0 = not check_env_bug(agent_actions[0], obs[0], agent_i=0, logging=logging)
                    if env_bug_count_a0:
                        print('Action Fail')
                        print('check_env_bug outside unity fail!')
                        valid_run_tem = False
                        break
        
                    all_cur_observation.append(deepcopy(obs))

                    # print('Action Success')
        
                    steps += 1
                    if np.any(dones):
                        valid_run_tem = True
                        if infos[0]['is_success']:
                            success_run_tem = True
                        break  # 结束动作执行循环
        
                # 动作执行结束，跳出验证循环
                break
            else:
                # 验证失败：增加验证计数，重新聚合世界模型后再次验证
                verify_attempt += 1
                print("Verification failed, attempt: {} out of {}".format(verify_attempt, max_verify_attempt))
                # 此处不更新成功率、可执行率等指标，直接继续循环以重新生成观察及计划
            # 如果这是倒数第二次验证尝试
                if verify_attempt == max_verify_attempt - 1:
                    # 检查是否有纠正计划
                    if memory and str(sample) in memory.corrected_plans and memory.corrected_plans[str(sample)].get("corrected_plan"):
                        # 获取纠正计划
                        corrected_plan = memory.corrected_plans[str(sample)]["corrected_plan"]
                        print("Using corrected plan on second-to-last attempt:")
                        print(corrected_plan)
                        
                        # 替换原始执行计划
                        llm_policy.exec_action_lists = corrected_plan.split('\n\n')
                        # 重置执行索引
                        llm_policy.exec_action_index = 0
                        
                        # 使用纠正计划进行验证
                        from llm_policy import de_final_complete_plan
                        from world_model import verify_action_plan
                        
                        # 保存当前的计划进行比较
                        original_de_final_complete_plan = de_final_complete_plan
                        # 更新全局计划变量以便验证
                        import llm_policy
                        llm_policy.de_final_complete_plan = corrected_plan
                        
                        # 使用新计划进行验证
                        verification_response, is_executable = verify_action_plan(merged_str, corrected_plan, memory=memory)
                        print(f"DEBUG: Verification result for corrected plan: is_executable = {is_executable}")
                        
                        # 无论验证结果如何，都使用纠正计划
                        # 不需要修改is_executable，让代码继续原有流程
                        continue        
            # 如果验证尝试次数超出最大次数，则直接进入下一个样本
            if verify_attempt >= max_verify_attempt:
                drop += 1
                memory.clear_memory_for_task(sample)
                print("Verification attempts exceeded, moving to new sample")
                continue

        # plan result
        print('*'*20,'Plan Results','*'*20)
        if valid_run_tem:
            valid_run += 1
            print('Executable Plan')

            if success_run_tem:
                success_count += 1
                print('Successful Plan')
            else:
                print('Plan is not successful')
                
        else:
            print('Plan is not executable')
        
            
        # increase task number
        i+=1

        if args.interactive_eval:
            execute_rate = 100. * valid_run / i if i != 0 else 0
            success_rate = 100. * success_count / i if i != 0 else 0
            print('*'*10,'Current Evaluation Metric','*'*10)
            print('Successful / Executable / Current / Total: {} / {} / {} / {}'.format(success_count,valid_run,i,len(testid)))
            print('Success Rate: {}'.format(success_rate))
            print('Executability: {}'.format(execute_rate))
            print('Now drop time is')
            print(drop)
            print('Attempt is: ')
            print(verify_attempt)
        sys.stdout.flush()

    print('*'*30,'Evaluation Metric on',args.subset,'*'*30)
    print('Successful / Executable / Total: {} / {} / {}'.format(success_count,valid_run,len(testid)))
    print('Success Rate: {}'.format(success_rate))
    print('Executability: {}'.format(execute_rate))
    
    return success_rate
