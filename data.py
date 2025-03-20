from datasets import load_dataset
from rich import print

prefix = """Online function calling is avalible while thinking.
function call format:
<function_call>
    <request>
    ...
    </request>
    <response>
    ...
    </response>
</function_call>
Available functions:

"""

class PicoThinkingFunctionCalling:
    def __init__(self):
        seed_dataset_name = "joey00072/pico_thinking_function_calling"
        self.seed_dataset = load_dataset(seed_dataset_name)["train"]
        def prepare(example):
            example["prompt"] = prefix+example["schema"]+"\n\n"+example["question"]
            example["tools"] = example["schema"]
            return example
        self.seed_dataset = self.seed_dataset.map(prepare)
        main_dataset_name = "Salesforce/xlam-function-calling-60k"
        self.main_dataset = load_dataset(main_dataset_name)["train"]
        def prepare(example):
            example["prompt"] = prefix+example["tools"]+"\n\n"+example["query"]
            return example
        self.main_dataset = self.main_dataset.map(prepare)
        
        
        self._seed_len = len(self.seed_dataset)
        self._main_len = len(self.main_dataset)
        
    def __len__(self):
        return self._seed_len + self._main_len
    
    def __iter__(self):
        for item in self.seed_dataset:
            yield item
        for item in self.main_dataset:
            yield item

    def __getitem__(self, idx):
        if idx < self._seed_len:
            return self.seed_dataset[idx]
        else:
            return self.main_dataset[idx - self._seed_len]
        
    def shuffle(self, seed=42):
        self.seed_dataset = self.seed_dataset.shuffle(seed=seed)
        self.main_dataset = self.main_dataset.shuffle(seed=seed)
        return self
    
    


if __name__ == "__main__":
    dataset = PicoThinkingFunctionCalling()

    for idx, item in enumerate(dataset):
        print(item)
        print("-"*100)
        if idx > 20:
            break
        
