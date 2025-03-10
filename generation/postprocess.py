import json
data_name = "coco"
file_nums = 1

group_summary_dict = dict()
meta_dict = dict()

if(file_nums>1):
    for i in range(file_nums):
        d = json.load(open(f"../data/{data_name}/annotations/{data_name}_group_summary_train.json.{i}","r"))
        group_summary_dict.update(d)
        meta_data = json.load(open(f"../data/{data_name}/annotations/{data_name}_chatgpt_train_group_summary.json.{i}","r"))
        meta_dict.update(meta_data)
else:
    group_summary_dict = json.load(open(f"../data/{data_name}/annotations/{data_name}_group_summary_train.json","r"))
    meta_dict = json.load(open(f"../data/{data_name}/annotations/{data_name}_chatgpt_train_group_summary.json","r"))

meta_keys = list(meta_dict.keys())
keys_list = list(group_summary_dict.keys())
n = 0
print("Total length: ", len(keys_list))
imgcoco2summary = {}
for k in keys_list:
    try:
        imgcoco2summary[k] = eval(group_summary_dict[k])
    except:
        n+=1
        continue
training_meta_w_group_id = []
summary_meta_w_group_id_for_diff = []
e = 0
for imgcoco_id,index_summary in imgcoco2summary.items():
    meta_list = meta_dict[imgcoco_id]
    
    try:
        index_k, summary_k = list(index_summary.keys())
        index = index_summary[index_k]
        summary = index_summary[summary_k]
        captions = []
        imgcoco_ids = []
        ann_temp_list = []
        for ii in index:
            ann = meta_list[int(ii)-1]
            captions.append(ann['caption'])
            imgcoco_ids.append(f"{ann['image_id']}_{ann['coco_id']}_{int(ii)}")
            ann.update({"group_id":imgcoco_id,"head_id":imgcoco_id}) # w_group_id 是用head_id 指示图像的
            ann_temp_list.append(ann)
        training_meta_w_group_id.extend(ann_temp_list)
        summary_meta_w_group_id_for_diff.append({"group_id":imgcoco_id,"caption":summary})
    except:
        print(index_summary)
print("len of summary meta for diff: ", len(summary_meta_w_group_id_for_diff))
print("len of training meta with group id: ", len(training_meta_w_group_id))

with open(f"../data/{data_name}/annotations/{data_name}_karpathy_train_w_group_id_chatgpt.json","w") as f:
    json.dump(training_meta_w_group_id,f)

with open(f"../data/{data_name}/annotations/{data_name}_karpathy_train_summary_chatgpt.json","w") as f:
    json.dump(summary_meta_w_group_id_for_diff,f)