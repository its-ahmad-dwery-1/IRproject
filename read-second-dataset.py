import json
filtered_json = []
counterId = 0
with open('corpus2/corpus.jsonl', 'r') as file:
    for line in file:
        j = json.loads(line)
        counterId = counterId + 1
        with open("corpus2/"+ str(counterId) +".text", 'w') as f:
            f.write(j[list(j.keys())[2]])