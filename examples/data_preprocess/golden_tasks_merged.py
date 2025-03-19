# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets
import json
import glob

from verl.utils.hdfs_io import copy, makedirs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='data/golden_tasks_merged')

    args = parser.parse_args()

    # Load all JSON files from the input directory
    json_files = glob.glob(os.path.join(args.input_dir, '*.json'))
    
    # Create a list to store all data
    all_data = []
    
    # Process each JSON file
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Keep all attributes from the original JSON
            all_data.append({
                "prompt": [{
                    "role": "user",
                    "content": data['user_messages'][0] + "\nThink through the problem step by step. Every time you want to execute a skill on the user's Excel file, specify the skill name and arguments in JSON format, enclosed in <SKILL>...</SKILL> tags, e.g. <SKILL>{\"name\": \"insert_column\", \"args\": {\"column_name\": \"E\"}}</SKILL>. The output of past skills will be provided to you in <OUTPUT>...</OUTPUT> tags immediately after </SKILL>, e.g. <SKILL>{\"name\": \"insert_column\", \"args\": {\"column_name\": \"E\"}}</SKILL><OUTPUT>{\"status\": \"success\", \"message\": \"Column E inserted successfully\"}</OUTPUT>.",
                }],
                "initialization": data['initialization'],
                "evaluation": data['evaluation'],
                "task_id": data['task_id'],
                "source": data['source'],
            })
    
    # Convert to datasets format
    dataset = datasets.Dataset.from_list(all_data)
    
    # Split into train (80%) and validation (20%) sets
    train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split['train']
    val_dataset = train_test_split['test']
    
    # Save as parquet files
    train_dataset.to_parquet(os.path.join(args.input_dir, 'train.parquet'))
    val_dataset.to_parquet(os.path.join(args.input_dir, 'val.parquet'))