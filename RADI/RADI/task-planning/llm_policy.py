import random
import time
import requests
import json
import re
import sys
import openai
import os
import torch
from sim_compute import Similarity
sys.path.append('FADI/FADI/virtualhome')
from simulation.unity_simulator import comm_unity as comm_unity
from transformers import GenerationConfig,AutoModelForCausalLM,AutoTokenizer,AutoModel
from peft import PeftModel
actionlist = []

def deduplicate_plan(plan_text: str) -> str:
# 按照连续空行将文本分割成块，每个块代表一组动作
    blocks = [block.strip() for block in plan_text.strip().split("\n\n") if block.strip()]
    seen = set()
    unique_blocks = []
    for block in blocks:
        if block not in seen:
            seen.add(block)
            unique_blocks.append(block)
    # 用两个换行符合并每个不重复的块
    return "\n\n".join(unique_blocks)

# generation config
generation_config = GenerationConfig(
        temperature=0.001,
        top_k=40,
        top_p=0.9,
        do_sample=True,
        num_beams=1,
        repetition_penalty=1.1,
        #max_new_tokens=32768
        #max_new_tokens=4096
        max_length=32768    
)

# split goal according to id
def split_goal(log, task_goal):
    # print('*'*20,'Current Step: Goal Decomposition','*'*20)
    id_dict=dict()
    items = task_goal.split(',')[:-1]
    for item in items:
        ids=re.findall('\(id:(\d+)\)',item)[0]
        if ids in id_dict:
            id_dict[ids].append(item)
        else:
            id_dict[ids] = [item]
    result_list = [','.join(group) for group in id_dict.values()]
    # print('*'*10,'Original Task Goal','*'*10)
    # print(task_goal)
    # print('*'*10,'Sub Task Goal','*'*10)
    # print(result_list)
    return result_list, len(result_list)

# def a class as a gpt policy
class LLMPolicy:
    def __init__(self, args, logging):
        
        if 'gpt' in args.llm:
            self.tokenizer=None
            self.llm=args.llm
        self.graph = None
        self.task_id=None
        self.task_goal = None
        self.split_task_goal = None
        self.split_task_goal_num = 0
        self.subtask=[]
        self.goal_exe_index = 0
        self.task_obj = []  #   ¼          ?   obj
        self.exec_action_lists = []
        self.exec_action_index = 0  # the index of the action to be executed
        self.goal_objs_loc = None
        self.logging = logging
        self.api=args.api
        self.mode=args.mode
        self.complete_plan=[]
        # use demo or not
        self.demo=args.demo
        if args.demo:
        # similarity module
            self.sc = Similarity()
            # action demo
            self.action_demo=[]
            self.action_match=[]
            data=json.load(open('FADI/FADI/task-planning/demo/action.json','r',encoding='utf-8'))
            for i in data:
                self.action_demo.append(i["content"])
                self.action_match.append(i["match"])
            # task demo
            self.task_demo=[]
            self.task_match=[]
            data=json.load(open('FADI/FADI/task-planning/demo/task.json','r',encoding='utf-8'))
            for i in data:
                self.task_demo.append(i["content"])
                self.task_match.append(i["match"])
            # react demo
            self.react_demo=[]
            self.react_match=[]
            data=json.load(open('FADI/FADI/task-planning/demo/react.json','r',encoding='utf-8'))
            for i in data:
                self.react_demo.append(i["content"])
                self.react_match.append(i["match"])
            # embodied demo
            self.embodied_demo=[]
            self.embodied_match=[]
            data=json.load(open('FADI/FADI/task-planning/demo/embodied.json','r',encoding='utf-8'))
            for i in data:
                self.embodied_demo.append(i["content"])
                self.embodied_match.append(i["match"])
            # goal-action demo
            self.goal_action_demo=[]
            self.goal_action_match=[]
            data=json.load(open('FADI/FADI/task-planning/demo/goal-action.json','r',encoding='utf-8'))
            for i in data:
                self.goal_action_demo.append(i["content"])
                self.goal_action_match.append(i["match"])   
            # task-action demo
            self.task_action_demo=[]
            self.task_action_match=[]
            data=json.load(open('FADI/FADI/task-planning/demo/task-action.json','r',encoding='utf-8'))
            for i in data:
                self.task_action_demo.append(i["content"])
                self.task_action_match.append(i["match"])   
            print("this time demo --------------")
            print(len(self.task_action_demo))

        # data collection
        # input and output of task decomposition
        self.task_prompt=[]
        # without demo
        self.task_res=[]
        # input and output of action decomposition
        self.action_prompt=[]
        # without demo
        self.action_res=[]
        # partial observation
        self.partial_locate=[]
        self.partial_state=[]
        self.exec_action_lists = []
        # 用于存储完整的 LLM 原始输出（完整计划）
    def get_corrected_plan_demo(self, task_or_goal):
        # 加载 WorldModelMemory
        try:
            from world_model import WorldModelMemory
            memory = WorldModelMemory(memory_file="worldmodel_memory.json")
            
            # 遍历所有保存的修正计划
            demo_plans = []
            for task_id, plan_data in memory.corrected_plans.items():
                if plan_data and "corrected_plan" in plan_data:
                    demo_plans.append(plan_data["corrected_plan"])
            
            # 如果没有找到任何计划，返回空字符串
            if not demo_plans:
                return ""
                
            # 构建参考示例字符串
            demo = ""
            
            # 选择最新的一个或两个计划作为参考
            recent_plans = demo_plans[-1:] if len(demo_plans) > 1 else demo_plans
            
            for i, plan in enumerate(recent_plans, 1):
                demo += f"# Reference Plan {i}:\n"
                for line in plan.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        demo += f"{line}\n"
                demo += "\n"
            
            return demo
            
        except Exception as e:
            print(f"获取修正计划示例时出错: {e}")
            return ""
    def reset(self,ids):
        self.graph = None
        self.task_id=ids
        self.task_goal = None
        self.split_task_goal = None
        self.split_task_goal_num = 0
        self.subtask=[]
        self.goal_exe_index = 0
        self.task_obj = []  #   ¼          ?   obj
        self.exec_action_lists = []
        self.exec_action_index = 0  # the index of the action to be executed
        self.goal_objs_loc = None
        self.complete_plan=[]
        # data collection
        # input and output of task decomposition
        self.task_prompt=[]
        # without demo
        self.task_res=[]
        # input and output of action decomposition
        self.action_prompt=[]
        # without demo
        self.action_res=[]
        # partial observation
        self.partial_locate=[]
        self.partial_state=[]
        
    def getLLMResponse(self,prompt,max_retries=1000):
        # open source llm
        if self.tokenizer:
            PermissionError
        
        # closed source llm
        else:
            # set openai key
            openai.api_key = self.api
            # adopt external interface
            # openai.api_base = "https://api.aigcbest.top/v1"
            
            # set retries
            retries=0
            while retries < max_retries:
                try:
                    # res = openai.ChatCompletion.create(
                    #     model=self.llm,
                    #     messages=[
                    #         {'role': 'user', 'content': prompt}
                    #     ],
                    #     temperature=0,
                    # )
                    # return res['choices'][0]['message']['content']
                    from test_openai import get_response
                    res = get_response(prompt, model=self.llm, temperature=0)
                    # from test_model import chat_with_model
                    # res = chat_with_model(prompt)
                    return res
                except Exception as e:
                    print(f"Exception caught: {e}")
                    retries += 1
                    time.sleep(5)
    
    def set_graph(self, graph):
        self.graph = graph

    def set_goal(self, lid_goals):
        task_goal = ''
        goal_objs = []  # Ŀ    Ʒ
        # translate the env_task_goal_write to natural language
        for k, v in lid_goals.items():
            if v > 0:
                obj_id = int(k.split('_')[-1])
                obj_name = [node['class_name'] for node in self.graph['nodes'] if node['id'] == obj_id][0]
                #  жϵ ǰobj Ƿ   goal   Ѵ  ڱ    ظ   ?
                have_exist_in_goal = False
                for id, name in goal_objs:
                    if id == obj_id:
                        have_exist_in_goal = True
                if not have_exist_in_goal:
                    goal_objs.append((obj_id, obj_name))
                #  жϵ ǰgoal_obj  task_obj   Ƿ  Ѵ ?
                have_exist = False
                for id, name in self.task_obj:
                    if id == obj_id and obj_name == name:
                        have_exist = True
                if not have_exist:
                    self.task_obj.append((obj_id, obj_name))
                task_goal += k.replace(k.split('_')[-1], obj_name + "(id:{})".format(obj_id)) + ': ' + str(v) + ','
                #   ȡ   obj
                name = str(k.split('_')[-2])
                for node in self.graph['nodes']:
                    if node['class_name'] == name:
                        goal_objs.append((node['id'], name))

        # print('[INFO] task goal GPT version:', task_goal)
        self.task_goal = task_goal
        # print('[INFO] goal obj:')
        # find the location of the goal objects
        goal_objs_loc = []  #    еص  Ŀ    Ʒ
        for obj_id, obj_name in goal_objs:
            from_obj_edges = [edge for edge in self.graph['edges'] if edge['from_id'] == obj_id]
            for edge in from_obj_edges:
                if edge['relation_type'] == 'INSIDE':
                    to_obj_id = edge['to_id']
                    to_obj_name = [node['class_name'] for node in self.graph['nodes'] if node['id'] == to_obj_id][0]
                    self.task_obj.append((to_obj_id, to_obj_name))
                    goal_objs_loc.append(('%s(id:%d)' % (obj_name, obj_id), edge['relation_type'],
                                          '%s(id:%d)' % (to_obj_name, to_obj_id)))

        self.goal_objs_loc = goal_objs_loc
        self.task_goal = task_goal

    def get_goal_obj_message(self, task):
        # closed_microwave(id:158): 1,turnon_microwave(id:158): 1,on_milk_kitchentable(id:123): 3,inside_pancake_microwave(id:158): 1,
        goals = task.split(',')
        need_grab_obj = []
        # list of object location
        goal_objs_loc = []
        # list of object state
        goal_objs_state = []
        need_put_obj = []
        need_get_obj = []
        reason_message = []
        for goal in goals:
            obj = goal.split('_')
            for name in obj:
                for node in self.graph['nodes']:
                    if node['class_name'] == name:
                        need_grab_obj.append((node['id'], name))
                pattern = r'(\w+)\(id:(\d+)\)'
                matches = re.findall(pattern, name)
                if matches:
                    id_ = int(matches[0][1])
                    name_ = matches[0][0]
                    id_list = [id_ for id_, name_ in need_put_obj]
                    if id_ not in id_list:
                        need_put_obj.append((id_, name_))

        for obj_id, obj_name in need_grab_obj:
            reason_message.append('%s(id:%d)' % (obj_name, obj_id))
            from_obj_edges = [edge for edge in self.graph['edges'] if edge['from_id'] == obj_id]
            for edge in from_obj_edges:
                #print(edge)
                #print(obj_id)
                #print(obj_name)
                #print(edge['to_id'])
                #print([node['class_name'] for node in self.graph['nodes'] if node['id'] == edge['to_id']])
                if edge['relation_type'] == 'INSIDE':
                    #print(edge)
                    to_obj_id = edge['to_id']
                    #print([node['class_name'] for node in self.graph['nodes'] if node['id'] == to_obj_id])
                    to_obj_name = [node['class_name'] for node in self.graph['nodes'] if node['id'] == to_obj_id][0]
                    goal_objs_loc.append(('%s(id:%d)' % (obj_name, obj_id), edge['relation_type'],
                                          '%s(id:%d)' % (to_obj_name, to_obj_id)))
                    id_list = [id_ for id_, name_ in need_get_obj]
                    if to_obj_id not in id_list:
                        need_get_obj.append((to_obj_id, to_obj_name))
                
                
        # get relevant object state
        obj_state = ''
        for obj_id, obj_name in need_put_obj:
            state = ''
            reason_message.append('%s(id:%d)' % (obj_name, obj_id))
            for node in self.graph['nodes']:
                if node['id'] == obj_id:
                    state = node['states']
                    break
            if state != '':
                if 'OPENED' in state and 'ON' in state:
                    obj_state += '{}(id:{}) is open|on, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','open|on'])
                    continue
                if 'OPENED' in state and 'OFF' in state:
                    obj_state += '{}(id:{}) is open|off, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','open|off'])
                    continue
                if 'CLOSED' in state and 'ON' in state:
                    obj_state += '{}(id:{}) is closed|on, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','closed|on'])
                    continue
                if 'CLOSED' in state and 'OFF' in state:
                    obj_state += '{}(id:{}) is closed|off, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','closed|off'])
                    continue
                if 'OPENED' in state:
                    obj_state += '{}(id:{}) is open, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','open'])
                    continue
                if 'CLOSED' in state:
                    obj_state += '{}(id:{}) is closed, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','closed'])
                    continue
                if 'ON' in state:
                    obj_state += '{}(id:{}) is on, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','on'])
                    continue
                if 'OFF' in state:
                    obj_state += '{}(id:{}) is off, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','off'])
                    continue
        
        for obj_id, obj_name in need_get_obj:
            state = ''
            for node in self.graph['nodes']:
                if node['id'] == obj_id:
                    state = node['states']
                    break
            if state != '':
                if 'OPENED' in state:
                    obj_state += '{}(id:{}) is open, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','open'])
                    continue
                if 'CLOSED' in state:
                    obj_state += '{}(id:{}) is closed, '.format(obj_name, obj_id)
                    goal_objs_state.append(['{}(id:{})'.format(obj_name, obj_id),'STATE','closed'])
                    continue

        obj_loc=''
        for i in goal_objs_loc:
            obj_loc=obj_loc+i[0]+' is in '+i[2]+', '
            
        # record goal_objs_location and goal_bojs_state
        self.partial_locate.append(goal_objs_loc)
        self.partial_state.append(goal_objs_state)
        
        # return all relevant obj
        obj=[]
        obj.extend(need_grab_obj)
        obj.extend(need_put_obj)
        obj.extend(need_get_obj)
        return obj_loc[:-2], obj_state[:-2], obj

        # return str(reason_message)

    def get_subtask_message(self, reason_subtask):
        pattern = r"id:(\d+)"
        ids = re.findall(pattern, reason_subtask)
        goal_objs_loc = []
        need_get_obj = []
        obj_state = ''
        for id_ in ids:
            id_ = int(id_)
            from_obj_edges = [edge for edge in self.graph['edges'] if edge['from_id'] == id_]
            to_obj_edges = [edge for edge in self.graph['edges'] if edge['to_id'] == id_]
            nodes = [node for node in self.graph['nodes'] if node['id'] == id_]
            if nodes:
                obj_name = nodes[0]['class_name']
            else:
                return False
            for edge in from_obj_edges:
                if edge['relation_type'] == 'INSIDE':
                    to_obj_id = edge['to_id']
                    to_obj_name = [node['class_name'] for node in self.graph['nodes'] if node['id'] == to_obj_id][0]
                    goal_objs_loc.append(('%s(id:%d)' % (obj_name, id_), edge['relation_type'],
                                          '%s(id:%d)' % (to_obj_name, to_obj_id)))
                    ids.append(to_obj_id)
                    #print(ids)
            state = ''
            for edge in to_obj_edges:
                if edge['relation_type'] == 'HOLDS_RH':
                    obj_state += '{}(id:{}) in your hand,'.format(obj_name, id_)
            for node in self.graph['nodes']:
                if node['id'] == id_:
                    state = node['states']
                    break
            if state != '':
                if 'OPENED' in state:
                    obj_state += '{}(id:{})\'s states are '.format(obj_name, id_)
                    obj_state += 'opened,'
                if 'CLOSED' in state:
                    obj_state += '{}(id:{})\'s states are '.format(obj_name, id_)
                    obj_state += 'closed,'
                if 'ON' in state:
                    obj_state += 'on,'
                if 'OFF' in state:
                    obj_state += 'off,'
        if obj_state:
            state_memory = str(list(set(goal_objs_loc))) + ' and ' + obj_state
        else:
            state_memory = str(list(set(goal_objs_loc)))
        return state_memory

    # get top k demonstration
    def get_demo(self, K, match, demo_list, match_list):
        # -----------------demo build---------------------------
        exampleTask = ''

        # Extract the string information of each task
        scores = []
        sim_score = self.sc.sim_compute(match,match_list)
        for index,demo in enumerate(demo_list):
            scores.append([sim_score[index], match_list[index], demo])

        scores.sort(reverse=True)

        topk = scores[:K]
        examplelist = [score[2] for score in topk]

        for demo in examplelist:
            exampleTask = exampleTask+demo.strip()+'\n\n'
        return exampleTask

    def generate_plan(self, task):
        global actionlist
        
        # print('*'*20,'Current Step: Generate plan for {}'.format(task),'*'*20)
        obs_loc,obs_state,obj=self.get_goal_obj_message(task)

        actionPrimitives = "from actions import " \
                           "walk <obj>, grab <obj>, switchon <obj>, " \
                           "open <obj>, close <obj>, " \
                           "putin <obj> <obj>, putback <obj> <obj>\n"
        reference_plan = self.get_corrected_plan_demo(task)
        if self.demo:
            mask_task=re.sub('\(id:\d+\)','',task) 
            if self.mode=='react': 
                exampleTask = self.get_demo(3, mask_task,self.react_demo,self.react_match)
            if self.mode=='goal-action':
                exampleTask = self.get_demo(3, mask_task,self.goal_action_demo,self.goal_action_match)
        else:
            exampleTask=''
        # version 3: without rule
        #full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        # version 2: with global action and rule
        full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'
        if reference_plan:
            full_prompt += reference_plan + '\n'        
        # version 3: each example with action and rule
        #full_prompt=exampleTask+actionPrimitives+rulePrompt+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        next_prompt = "# task goal: " + task + "\ndef task():\n"

        final_prompt = full_prompt + next_prompt
        context = self.getLLMResponse(final_prompt)
        self.complete_plan.append({
            "type": "task_decomposition",
            "prompt": final_prompt.strip(),
            "response": context.strip()
        })
        # process context for it may have more than one sample
        if '# key object location:' in context:
            context=context.split('# key object location:')[0]
        # print('*'*10,'Prompt','*'*10)
        # print(final_prompt)
        # print('*'*10,'LLM Output','*'*10)
        # print("1")
        # print(context)
        self.context_analysis(context)
        self.task_prompt.append([final_prompt.strip(), context.strip()])
        self.task_res.append([actionPrimitives+'\n'+'# key object location: '+obs_loc+'\n'+next_prompt.strip(), context.strip()])
    # MLDT
    def generate_multi_layer_plan(self):
        global final_complete_plan
        global de_final_complete_plan
        # goal-level decomposition
        self.split_task_goal, self.split_task_goal_num = split_goal(self.logging, self.task_goal)
        # task-level decomposition
        for goal in self.split_task_goal:
            self.split_task(goal)
            
        # action-level decomposition
        for task in self.subtask:
            self.generate_subplan(task)
        
        
        # print('*'*20,'Action List','*'*20)
        for action in self.exec_action_lists:
            
            actionlist.append(action)
            
            # print(action)
        #print("len exec_action_lists",len(self.exec_action_lists))
        #print("len complete plan",len(self.complete_plan))
        final_complete_plan = "\n\n".join(self.exec_action_lists)
        #final_complete_plan = "\n\n".join([entry["response"] for entry in self.complete_plan])
        #print("complete plan", self.complete_plan)
        #print("final_complete_plan",final_complete_plan)
        de_final_complete_plan =deduplicate_plan(final_complete_plan)      
        print("de_final_complete_plan",de_final_complete_plan)
        
        # print('*' * 20, 'Final Complete Plan', '*' * 20)
        # print(final_complete_plan)
        # task level decomposition
    def split_task(self,goal):
 #       print('*'*20,'Current Step: Task Decomposition for {}'.format(goal),'*'*20)                 
        # action
        #actionPrimitives = "from action import grab <obj> in <obj>, put <obj> in <obj>, put <obj> on <obj>, switch on <obj>\n"
        actionPrimitives="Available actions: grab <obj> in <obj>, put <obj> in <obj>, put <obj> on <obj>, switch on <obj>. Below are three demos. DO NOT repeat or explain anything. Only output the final task in exactly the same format, starting immediately after the last 'def task():'."
        
        exampleTask=''
        # demo
        if self.demo:
            mask_goal=re.sub('\(id:\d+\)','',goal)  
            if self.mode=='multi-layer':
                exampleTask = self.get_demo(3,mask_goal,self.task_demo,self.task_match)
  #              print("split task","multi-layer")
   #             print(exampleTask)
            if self.mode=='task-action':
                exampleTask = self.get_demo(3,mask_goal,self.task_action_demo,self.task_action_match)
        reference_plan = self.get_corrected_plan_demo(goal)
        # partial observation
        obs_loc,obs_state,obj=self.get_goal_obj_message(goal)
        # prompt construct
        full_prompt=actionPrimitives+'\n'+'here are three demos'+exampleTask+'Now this task is:\n'+'# key object location: '+obs_loc+'\n'
        if reference_plan:
            full_prompt +='Here is a reference plan you can use as inspiration:'+reference_plan+ '\n'
        next_prompt = "# task goal: " + goal + "\ndef task():\n"
        final_prompt = full_prompt + next_prompt
        context = self.getLLMResponse(final_prompt)

    #    print("split task final prompt")
     #   print(final_prompt)
      #  print("--------------------answer")
       # print(context)
        if not context.strip():
            context = "No response generated."
        # process context for it may have more than one sample
        if '# key object location:' in context:
            context=context.split('# key object location:')[0]
        #print("split task context",context)
        # print('*'*10,'Prompt','*'*10)
        # print(final_prompt)
        # print('*'*10,'LLM Output','*'*10)
        # print("2")
        # print(context)
        # analyze the context
        # if '```' in context:
        #     context=context.split('```')[0]
        for line in context.split('\n'):
            line=line.strip()
            
            if len(line)!=0 and not line.startswith('#') and line.split(' ')[0] in ['grab','put','switch']:
                match1=re.findall('(grab|put) (.+) (in|on) (.+)',line)
                match2=re.findall('switch on (.+)',line)
                if match1 or match2:
                    self.subtask.append(line)
        #print("subtask",self.subtask)
            
            # if len(line)!=0 and not line.startswith('#'):
            #    self.subtask.append(line)
        # record 
        self.task_prompt.append([final_prompt.strip(),context.strip()])
        self.task_res.append([actionPrimitives+'\n'+'# key object location: '+obs_loc+'\n'+next_prompt.strip(),context.strip()])
       # print("task prompt",self.task_prompt)
       # print("task res",self.task_res)
    
    # action level decomposition: generate plan for subtask 
    def generate_subplan(self,task):
        # print('*'*20,'Current Step: Generate plan for {}'.format(task),'*'*20)
        actionPrimitives = "from actions import " \
                           "walk <obj>, grab <obj>, switchon <obj>, " \
                           "open <obj>, close <obj>, " \
                           "putin <obj> <obj>, putback <obj> <obj>\n"
        
        exampleTask=''
        # demo
        reference_plan = self.get_corrected_plan_demo(task)
        if self.demo:
            # mask obj task
            obj = re.findall(r'(grab|put) (.+) (in|on)', task)
            if obj:
                mask_task=task.replace(obj[0][1],'something')
            else:
                mask_task=task
            mask_task=re.sub('\(id:\d+\)','',mask_task)
            if self.mode=='multi-layer' or self.mode=='task-action':
                exampleTask = self.get_demo(3,mask_task,self.action_demo,self.action_match)
         #       print("generate subplan, i.e. action, mmulti-layer")
        #        print(exampleTask)

        # version 3: without rule
        #full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        # version 2: with global action and rule
        full_prompt=actionPrimitives+'\n'+exampleTask
        if reference_plan:
            full_prompt += 'Here is a reference plan you can use as inspiration:'+reference_plan + '\n'
        # version 3: each example with action and rule
        #full_prompt=exampleTask+actionPrimitives+rulePrompt+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        next_prompt = "# task: " + task + "\ndef task():\n"
        final_prompt = full_prompt + next_prompt
        context = self.getLLMResponse(final_prompt)
        #print("generate subplan, action, context",context)
        if not context.strip():
            context = "No response generated."
        self.complete_plan.append({
            "type": "task_decomposition",  # 或 "action_decomposition"
            "prompt": final_prompt.strip(),
            "response": context.strip()
        })
        self.complete_plan.append({
            "type": "action_decomposition",
            "prompt": final_prompt.strip(),
            "response": context.strip()
        })
        # process context for it may have more than one sample
        if '# task:' in context:
            context=context.split('# task:')[0]
       # print("subplan, action, context split",context)
        # print('*'*10,'Prompt','*'*10)
        # print(final_prompt)
        # print('*'*10,'LLM Output','*'*10)
        # print("3")
        # print(context)
        self.context_analysis(context)
        # record 
        self.action_prompt.append([final_prompt.strip(),context.strip()])
        self.action_res.append([actionPrimitives+'\n'+next_prompt.strip(),context.strip()])
              

    def context_analysis(self, context):
        lines = context.split('\n')
        id_list = [] 
        for line in lines:
            line=line.replace(" ", "")
            pattern = r"(walk|find|open|grab|close|switchon)\('(\w+)\(id:(\d+)\)'\)"
            match = re.match(pattern, line)
            if match:
                action = match.group(1)
                if action == 'find':
                    action = 'walk'
                item_name = match.group(2)
                item_id = match.group(3)
                action_script = "[{}] <{}> ({})".format(action, item_name, item_id)
                self.exec_action_lists.append(action_script)
            pattern = r"(putback|putin)\('(\w+)\(id:(\d+)\)','(\w+)\(id:(\d+)\)'\)"
            match = re.match(pattern, line)
            if match:
                action = match.group(1)
                item1_name = match.group(2)
                item1_id = match.group(3)
                item2_name = match.group(4)
                item2_id = match.group(5)
                action_script = "[{}] <{}> ({}) <{}> ({})".format(action, item1_name, item1_id, item2_name, item2_id)
                self.exec_action_lists.append(action_script)
    # generate plan for embodied prompt
    def generate_embodied(self, task):
        #print('*'*20,'Current Step: Generate plan for {}'.format(task),'*'*20)
        obs_loc,obs_state,obj=self.get_goal_obj_message(task)

        actionPrimitives = "from actions import " \
                           "walk <obj>, grab <obj>, switchon <obj>, " \
                           "open <obj>, close <obj>, " \
                           "putin <obj> <obj>, putback <obj> <obj>\n"
        if self.demo:
            mask_task=re.sub('\(id:\d+\)','',task)
            if self.mode=='embodied':
                exampleTask = self.get_demo(3, mask_task,self.embodied_demo,self.embodied_match)
        else:
            exampleTask=''
        # version 3: without rule
        #full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        # version 2: with global action and rule
        full_prompt=actionPrimitives+'\n'+exampleTask+'# key object location: '+obs_loc+'\n'
        # version 3: each example with action and rule
        #full_prompt=exampleTask+actionPrimitives+rulePrompt+'# key object location: '+obs_loc+'\n'+'# key object state: '+obs_state+'\n'
        next_prompt = "# task goal: " + task + "\ndef task():\n"
        final_prompt = full_prompt + next_prompt
        context = self.getLLMResponse(final_prompt)
        # process context for it may have more than one sample
        if '# key object location:' in context:
            context=context.split('# key object location:')[0]
        #print('*'*10,'Prompt','*'*10)
        
        #print(final_prompt)
        #print('*'*10,'LLM Output','*'*10)
        #print("4")
        #print(context)
        self.context_analysis(context)
    
    def rule_modify(self,max_loop=100):
        # judge satisfy rule or not
        FLAG=False
        
        # open/close object
        oc_obj=['cabinet','dishwasher','fridge','kitchencabinet','microwave','stove']
        
        # putin object
        putin_obj=oc_obj
        
        # putback object
        putback_obj=['sink','kitchentable']
        
        # switchon object
        switchon_obj=['stove','microwave','dishwasher']
        
        # set loop to avoid eternally loop
        loop=0
        
        # modify until not error occur
        while not FLAG and loop<max_loop: 
            FLAG=True
            loop+=1
            for index,i in enumerate(self.exec_action_lists):
                # match putin/putback
                matches = re.findall(r"\[(\w+)\] <(\w+)> \((\d+)\) <(\w+)> \((\d+)\)", i)
                if matches:
                    action=matches[0][0]
                    obj1=matches[0][1]
                    num1=matches[0][2]
                    obj2=matches[0][3]
                    num2=matches[0][4]
                else:
                    # match other actions
                    matches = re.findall(r"\[(\w+)\] <(\w+)> \((\d+)\)", i)
                    if matches:
                        action=matches[0][0]
                        obj=matches[0][1]
                        num=matches[0][2]
                
                # action after walk oc_obj must be open
                if action=='walk' and obj in oc_obj:
                    # gd
                    gd=i.replace('walk','open')
                    # not satisfy
                    if len(self.exec_action_lists)<=index+1:
                        self.exec_action_lists.append(gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index+1]!=gd:
                        self.exec_action_lists.insert(index+1,gd)
                        FLAG=False
                        break
                
                # action before grab must be walk, action after grab oc_obj must be close
                if action=='grab':
                    # gd before
                    gd=i.replace('grab','walk')
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break
                    
                    # if interact object is in oc_obj
                    # extract object from previous step
                    if index>=2:
                        pre_step=self.exec_action_lists[index-2]
                        matches = re.findall(r"\[(\w+)\] <(\w+)> \((\d+)\)", pre_step)
                        if matches:
                            pre_act=matches[0][0]
                            pre_obj=matches[0][1]
                            pre_num=matches[0][2]
                            if pre_act=='open' and pre_obj in oc_obj:
                                # gd after
                                gd=pre_step.replace('open','close')
                                # not satisfy
                                if len(self.exec_action_lists)<=index+1:
                                    self.exec_action_lists.append(gd)
                                    FLAG=False
                                    break
                                if self.exec_action_lists[index+1]!=gd:
                                    self.exec_action_lists.insert(index+1,gd)
                                    FLAG=False
                                    break
                
                # action before switchon must be close
                if action=='switchon':
                    # check target object restriction
                    if obj not in switchon_obj:
                        self.exec_action_lists.remove(i)
                        FLAG=False
                        break
                    # gd before
                    gd=i.replace('switchon','close')
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break
                
                # action before putin must be open, after must be close
                if action=='putin':
                    # check target object restriction
                    if obj2 not in putin_obj:
                        if obj2 in putback_obj:
                            self.exec_action_lists[index]=i.replace('putin','putback')
                            FLAG=False
                            break
                        else:
                            self.exec_action_lists.remove(i)
                            FLAG=False
                            break
                    # action before
                    gd="[{}] <{}> ({})".format('open', obj2, num2)
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break
                    # action after
                    gd="[{}] <{}> ({})".format('close', obj2, num2)
                    # not satisfy
                    if len(self.exec_action_lists)<=index+1:
                        self.exec_action_lists.append(gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index+1]!=gd:
                        self.exec_action_lists.insert(index+1,gd)
                        FLAG=False
                        break            
                
                # action before putback must be walk
                if action=='putback':
                    # check target object restriction
                    if obj2 not in putback_obj:
                        if obj2 in putin_obj:
                            self.exec_action_lists[index]=i.replace('putback','putin')
                            FLAG=False
                            break
                        else:
                            self.exec_action_lists.remove(i)
                            FLAG=False
                            break                                              
                    # action before
                    gd="[{}] <{}> ({})".format('walk', obj2, num2)
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break        
                
                # target object restriction of open/close
                if action in ['open','close']:
                    if obj not in oc_obj:
                        self.exec_action_lists.remove(i)      
                        FLAG=False
                        break
                
                # action before open must be walk
                if action=='open':
                    # action before
                    gd="[{}] <{}> ({})".format('walk', obj, num)
                    # not satisfy
                    if index==0:
                        self.exec_action_lists.insert(0,gd)
                        FLAG=False
                        break
                    if self.exec_action_lists[index-1]!=gd:
                        self.exec_action_lists.insert(index,gd)
                        FLAG=False
                        break
    
    def get_action_from_llm(self):
        action_obj_str = ''
        print("overall action length",len(self.exec_action_lists))
        if self.exec_action_index >= len(self.exec_action_lists):
            return 'DONE'
        action_obj_str = self.exec_action_lists[self.exec_action_index]
        self.exec_action_index += 1
        return action_obj_str


