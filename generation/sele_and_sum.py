import openai
import asyncio
import concurrent.futures
import json
import os
from tqdm import tqdm
import time
import aiohttp
import numpy as np

async def send_request(message, model="gpt-3.5-turbo", max_retries=2, retry_wait=1):
    url = "https://api.openai.com/v1/chat/completions"
    keys = [
        "sk-",
    ]
    
    data = {
        "messages": [{"role": "user", "content": message}],
        "model": model,
        "max_tokens": 500,
    }
    retries = 0
    while retries < max_retries:
        try:
            k = list(np.random.choice(keys,1))[0]
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {k}"
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, data=json.dumps(data)) as response:
                    response_data = await response.text()
            return json.loads(response_data)
        except Exception as e:
            retries += 1
            print(f"Error occurred. Retrying {retries}/{max_retries}... Error: {e}")
            if retries < max_retries:
                await asyncio.sleep(retry_wait)
            else:
                print(f"Failed after {max_retries} retries.")
                return None


async def send_multiple_requests(image_ids,prompts):
    tasks = [send_request(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)
    response_ids_contents = {}
    for i, response in enumerate(responses):
        if(response is not None):
            try:
                response_ids_contents[image_ids[i]] = response["choices"][0]["message"]["content"]
            except:
                response_ids_contents[image_ids[i]] = None
                print(response)
        else:
            response_ids_contents[image_ids[i]] = None
    return response_ids_contents

async def main():
    task = "group_summary"
    task_prompt = "Select and summary sentences in the given sentences set. You should find 5 to 10 sentences that form a description from the same or different views of the same image. The meanings of the selected sentences in a group should not conflict with each other. Summarize the sentences as one not exceed 50 words to describe the scene in objective style. The summary sentence must be objective and concise, without ambiguity and uncertainty. Return the selected index and the summary sentence in json format like {'index': list,'summary': str}. Return directly the json format results without explanation. The given sentences set are: "
    
    split = "train"
    data_name = "coco"
    file_name = f"../data/{data_name}/annotations/{data_name}_chatgpt_{split}_group_summary.json"
    with open(file_name,"r") as f:
        coco_caption_dict = json.load(f)

    image_ids = list(coco_caption_dict.keys()) # [:20]
    for img_id in image_ids:
        captions = []
        ann_list = coco_caption_dict[img_id]
        for ann in ann_list:
            c = ann['caption']
            if(c.endswith(".")):
                captions.append(c)
            else:
                captions.append(c+".")
        coco_caption_dict[img_id] = captions
    chatgpt_response_json = f"../data/{data_name}/annotations/{data_name}_{task}_{split}.json"
    if(os.path.isfile(chatgpt_response_json)):
        chatgpt_response_dict = json.load(open(chatgpt_response_json,"r"))
        new_chatgpt_response_dict = {}
        for k,v in chatgpt_response_dict.items():
            if(v is not None):
                new_chatgpt_response_dict[k] = v
        chatgpt_response_dict = new_chatgpt_response_dict
    else:
        chatgpt_response_dict = {}
    chatgpt_response_dict['prompt'] = task_prompt
    
    prompts = []
    new_image_ids = []
    for image_id in image_ids:
        if(image_id in chatgpt_response_dict.keys()):
            continue
        new_image_ids.append(image_id)
        captions = coco_caption_dict[image_id]
        s = ""
        for i,c in enumerate(captions):
            c+=" "
            index = f"<{i+1}> "
            s += index + c
        prompts.append(task_prompt+s)
    image_ids = new_image_ids
    batch_size = 100
    sleep_interval = 1
    num_batches = len(prompts) // batch_size + (1 if len(prompts) % batch_size else 0)
    for i in tqdm(range(num_batches)):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        batch_prompts = prompts[start_index:end_index]
        batch_img_ids = image_ids[start_index:end_index]
        response_ids_contents = await send_multiple_requests(batch_img_ids,batch_prompts)
        chatgpt_response_dict.update(response_ids_contents)
        with open(chatgpt_response_json,"w") as f:
            json.dump(chatgpt_response_dict,f)
        if i < num_batches - 1:
            time.sleep(sleep_interval)
if __name__ == "__main__":
    asyncio.run(main())
