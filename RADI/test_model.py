import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 或你有的 GPU 编号
import sys
print(">>> Python interpreter:", sys.executable)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("now device",device)

model_path = './autodl-tmp/models/sft' 
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
    local_files_only=True  # 防止尝试连接 HuggingFace Hub
)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
).to(device)

# 打印模型设备
print(f"Model is on device: {next(model.parameters()).device}")

# def chat_with_model(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     # 打印输入张量设备
#     print(f"Input tensor is on device: {inputs['input_ids'].device}")
    
#     outputs = model.generate(
#         inputs["input_ids"],
#         max_length=2048,
#         num_return_sequences=1,
#         no_repeat_ngram_size=2
#        # eos_token_id=tokenizer.eos_token_id 
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

chat_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def chat_with_model(messages, max_new_tokens=2048):
    """
    与 LLaMA-3-Instruct 本地模型进行对话，输入为 messages 列表，输出为生成文本。
    messages: [{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]
    """
    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_token_ids = [tokenizer.eos_token_id, eot_token_id]

    outputs = chat_pipeline(
        messages,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=eos_token_ids
    )

    return outputs[0]["generated_text"]

# prompt="""Available actions: grab <obj> in <obj>, put <obj> in <obj>, put <obj> on <obj>, switch on <obj>. Below are three demos. DO NOT repeat or explain anything. Only output the final task in exactly the same format, starting immediately after the last 'def task():'.
# here are three demos# key object location: poundcake(id:339) is in cabinet(id:222), poundcake(id:340) is in cabinet(id:222)
# # task goal: closed_fridge(id:149): 1,inside_poundcake_fridge(id:149): 2
# def task():
# # the goal means the task is "put two poundcakes in fridge"
# # grab the first poundcake
# grab poundcake(id:339) in cabinet(id:222)
# # put the first poundcake in fridge
# put poundcake(id:339) in fridge(id:149)
# # grab the second poundcake
# grab poundcake(id:340) in cabinet(id:222)
# # put the second poundcake in fridge
# put poundcake(id:340) in fridge(id:149)

# # key object location: poundcake(id:333) is in cabinet(id:222), poundcake(id:334) is in cabinet(id:222)
# # task goal: closed_fridge(id:149): 1,inside_poundcake_fridge(id:149): 2
# def task():
# # the goal means the task is "put two poundcakes in fridge"
# # grab the first poundcake
# grab poundcake(id:333) in cabinet(id:222)
# # put the first poundcake in fridge
# put poundcake(id:333) in fridge(id:149)
# # grab the second poundcake
# grab poundcake(id:334) in cabinet(id:222)
# # put the second poundcake in fridge
# put poundcake(id:334) in fridge(id:149)

# # key object location: poundcake(id:332) is in stove(id:150), poundcake(id:333) is in kitchen(id:50), poundcake(id:339) is in cabinet(id:222)
# # task goal: closed_fridge(id:149): 1,inside_poundcake_fridge(id:149): 2
# def task():
# # the goal means the task is "put two poundcakes in fridge"
# # grab the first poundcake
# grab poundcake(id:332) in stove(id:150)
# # put the first poundcake in fridge
# put poundcake(id:332) in fridge(id:149)
# # grab the second poundcake
# grab poundcake(id:333) in kitchen(id:50)
# # put the second poundcake in fridge
# put poundcake(id:333) in fridge(id:149)

# Now this task is:
# # key object location: poundcake(id:332) is in stove(id:150), poundcake(id:333) is in kitchen(id:50)
# # task goal: closed_fridge(id:149): 1,inside_poundcake_fridge(id:149): 2
# def task():"""

# print(chat_with_model(prompt))
