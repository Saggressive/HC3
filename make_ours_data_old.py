#从测试集里划分验证集
import json
import csv
from datasets import load_dataset
import os
import random

def check_contain_chinese(check_str):
    for ch in check_str.encode('utf-8'). decode('utf-8'):
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def proc_data(files,mode="train",language="en"):
    all_proc_data = []
    for file in files:
        with open(file,"r") as f:
            data = f.readlines()
            data = [ json.loads(i) for i in data ]
            if mode=="test" and "hc3_full_json" not in file and "hc3_sent_json" not in file:
                random.shuffle(data)
                data=data[:3000]
            for row in data:
                if "label" in row.keys():
                    row = json.dumps(row,ensure_ascii=False)
                    row = row+"\n"
                    all_proc_data.append(row)
                else:
                    if "source" in row.keys() or "article" in row.keys():
                        source_text = row["source"] if "source" in row.keys() else row["article"]
                        source_text = source_text.strip()
                        if source_text == "":
                            continue
                        if language=="zh" and "article" in row.keys():
                            have_chinese = check_contain_chinese(source_text)
                            if have_chinese==False:
                                continue
                    gpt_gen = row["gpt_gen"]
                    human = row["target"]
                    #简单去重
                    if gpt_gen.strip() == human.strip():
                        continue
                    proc_row = {
                        "text":gpt_gen,
                        "label":1
                    }
                    proc_row = json.dumps(proc_row,ensure_ascii=False)
                    proc_row = proc_row+"\n"    
                    all_proc_data.append(proc_row)

                    proc_row = {
                        "text":human,
                        "label":0
                    }
                    proc_row = json.dumps(proc_row,ensure_ascii=False)
                    proc_row = proc_row+"\n"     
                    all_proc_data.append(proc_row)
    if mode == "test":
        random.shuffle(all_proc_data)
        length = len(all_proc_data)
        split = int(len(all_proc_data)*0.5)
        test_data,val_data = all_proc_data[0:split],all_proc_data[split:]
        return test_data,val_data
    elif mode == "train":
        return all_proc_data

def make_ours_train_data(save_folder,language):
    root = save_folder + os.sep + language
    os.makedirs(root,exist_ok=True)
    if language == "en":
        ours_files = [
            "clean_null_data_train/cnndm_chatgpt/train.jsonl",
            "clean_null_data_train/en_de/de2en_train.jsonl",
            "clean_null_data_train/en_fr/fr2en_train.jsonl",
            "clean_null_data_train/en_ro/ro2en_train.jsonl",
            "clean_null_data_train/en_zh/zh2en_train.jsonl",
            "clean_null_data_train/hc3_chatgpt_question/en_train.jsonl",
            "clean_null_data_train/xsum_chatgpt/train.jsonl"
        ]
        hc3_files = ["clean_null_data_train/hc3_full_json/en_train.jsonl"]
    elif language=="zh":
        ours_files = [
            "clean_null_data_train/en_zh/en2zh_train.jsonl",
            "clean_null_data_train/hc3_chatgpt_question/zh_train.jsonl",
            "clean_null_data_train/lcsts_chatgpt/train.jsonl",
            "clean_null_data_train/news2016_chatgpt/train.jsonl",
        ]
        hc3_files = ["clean_null_data_train/hc3_full_json/zh_train.jsonl"]
    train_save_path = save_folder + os.sep + language + os.sep +  "train.jsonl"
    # val_ours_save_path = save_folder + os.sep + language + os.sep +  "val_ours.jsonl"
    # val_hc3_save_path = save_folder + os.sep + language + os.sep +  "val_hc3.jsonl"
    ours_train_data = proc_data(ours_files,"train",language=language)
    hc3_train_data = proc_data(hc3_files,"train",language=language)
    train_data = ours_train_data + hc3_train_data
    with open(train_save_path,"w") as f:
        f.writelines(train_data)
    # with open(val_ours_save_path,"w") as f:
    #     f.writelines(ours_val_data)
    # with open(val_hc3_save_path,"w") as f:
    #     f.writelines(hc3_val_data)

def make_ours_test_data(save_folder,language):
    root = save_folder + os.sep + language
    os.makedirs(root,exist_ok=True)
    if language == "en":
        ours_files = [
            "clean_null_data_train/hc3_chatgpt_question/en_test.jsonl",
            "clean_null_data_val/cnndm_chatgpt_clean/val.jsonl",
            "clean_null_data_val/en_de_clean/de2en.jsonl",
            "clean_null_data_val/en_fr_clean/fr2en.jsonl",
            "clean_null_data_val/en_ro_clean/ro2en.jsonl",
            "clean_null_data_val/en_zh_clean/zh2en.jsonl",
            "clean_null_data_val/xsum_chatgpt_clean/val.jsonl",
        ]
        hc3_files = ["clean_null_data_train/hc3_full_json/en_test.jsonl"]
    elif language=="zh":
        ours_files = [
            "clean_null_data_train/hc3_chatgpt_question/zh_test.jsonl",
            "clean_null_data_val/lcsts_chatgpt_clean/val.jsonl",
            "clean_null_data_val/news2016_chatgpt_clean/val.jsonl",
            "clean_null_data_val/en_zh_clean/en2zh.jsonl"
        ]
        hc3_files = ["clean_null_data_train/hc3_full_json/zh_test.jsonl"]
    test_ours_save_path = save_folder + os.sep + language + os.sep +  "test_ours.jsonl"
    test_hc3_save_path = save_folder + os.sep + language + os.sep +  "test_hc3.jsonl"
    val_ours_save_path = save_folder + os.sep + language + os.sep +  "val_ours.jsonl"
    val_hc3_save_path = save_folder + os.sep + language + os.sep +  "val_hc3.jsonl"

    ours_test_data,ours_val_data = proc_data(ours_files,"test",language)
    hc3_test_data,hc3_val_data = proc_data(hc3_files,"test",language)

    with open(test_ours_save_path,"w") as f:
        f.writelines(ours_test_data)
    with open(test_hc3_save_path,"w") as f:
        f.writelines(hc3_test_data)
    with open(val_ours_save_path,"w") as f:
        f.writelines(ours_val_data)
    with open(val_hc3_save_path,"w") as f:
        f.writelines(hc3_val_data)

if __name__=="__main__":
    #v1没划分我们的数据和对方的数据，v2没有做去重，v3去除gpt和人类一致的部分
    save_folder="ours_dataset_v3"
    language="zh"
    make_ours_train_data(save_folder,language)
    make_ours_test_data(save_folder,language)