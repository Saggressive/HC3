import json
from datasets import load_dataset
import os
def convert_csv_to_json(folder_path,save_folder_path):
    os.makedirs(save_folder_path,exist_ok=True)
    files = os.listdir(folder_path)

    for file in files:
        file_path = folder_path + os.sep + file
        dataset = load_dataset("csv",data_files=file_path)["train"]
        proc_row = []
        for row in dataset:
            answer = row["answer"]
            label = row["label"]
            text = {"text":answer,"label":label}
            text = json.dumps(text,ensure_ascii=False)
            text = text+"\n"
            proc_row.append(text)

        save_path = save_folder_path + os.sep + file.replace("csv", "jsonl")
        with open(save_path,"w") as f:
            f.writelines(proc_row)

if __name__ == "__main__":
    folder_path = "hc3_full"
    save_folder_path = "hc3_full_json"
    convert_csv_to_json(folder_path,save_folder_path)