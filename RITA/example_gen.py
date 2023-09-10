import torch, os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from tokenizers import Tokenizer
from RITA_s.rita_modeling import RITAModelForCausalLM


def get_tokenizer(file="seq-1M/tokenizer.json"):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())

device = "cuda"

model = RITAModelForCausalLM.from_pretrained("seq-1M/").to(device)
tokenizer = get_tokenizer()



context = "PDALALAGSSGSSGVMVFISSSLNSFRSEKRYSRSLTIAEFKCKLELVVGSPASCMELELYGADDKFYSKLDQEDALLGSYPVDDGCRIHVIDHSGSGPSSG"
temp = 0.8
max_length = 128
top_p = 0.9
num_return_sequences = 2
pad_token_id = 2

with torch.no_grad():
    input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
    output = model(input_ids, labels=input_ids)
    print(output.loss.item())
    

with torch.no_grad():
    input_ids = torch.tensor(tokenizer.encode(context).ids).view([1, -1]).to(device)
    tokens_batch = model.generate(input_ids, 
                                  do_sample=True, 
                                  temperature=temp, 
                                  max_length=max_length, top_p=top_p, num_return_sequences=num_return_sequences, pad_token_id=pad_token_id)
    as_lists = lambda batch: [batch[i, ...].detach().cpu().numpy().tolist() for i in range(batch.shape[0])]

print(tokenizer.decode_batch(as_lists(tokens_batch)))