import json

if __name__ == '__main__':
    with open('output/_data1.json') as f:
        data1 = json.load(f)
    with open('output/_data2.json') as f:
        data2 = json.load(f)
    data = data1 + data2
    with open('output/data.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False)

    elapsed_time = 0
    input_tokens = 0
    output_tokens = 0
    llm_calls = 0
    for item in data:
        elapsed_time += item['elapsed_time']
        input_tokens += item['input_tokens']
        output_tokens += item['output_tokens']
        llm_calls += item['llm_calls']


    # creating metadata CSVs
    with open('output/metadata_general.csv', 'w') as f:
        f.write('elapsed_time,input_tokens,output_tokens,llm_calls\n')
        f.write(f'{elapsed_time},{input_tokens},{output_tokens},{llm_calls}\n')

    # creating metadata CSVs for each item
    with open('output/metadata_per_text.csv', 'w') as f:
        f.write('elapsed_time,input_tokens,output_tokens,llm_calls\n')
        for item in data:
            f.write(f'{item["elapsed_time"]},{item["input_tokens"]},{item["output_tokens"]},{item["llm_calls"]}\n')
