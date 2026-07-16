from simplificacao import LinguagemSimplesGraph
import json
import time
import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simplificação de textos jurídicos")
    parser.add_argument("--model", type=str, required=True, help="Nome do modelo a ser utilizado")
    args = parser.parse_args()

    # create output directory if it doesn't exist
    os.makedirs(f"output/{args.model}", exist_ok=True)

    # import input/corpus.json
    with open("input/corpus.json", "r") as f:
        corpus = json.load(f)
    with open(f"output/{args.model}/metadata_per_text.csv", "w") as f:
        f.write("elapsed_time,input_tokens,output_tokens,llm_calls\n")

    total_texts = len(corpus)
    processed_texts = 1
    total_input_tokens = 0
    total_output_tokens = 0
    total_time_elapsed = 0
    llm_calls = 0

    for text in corpus:
        print(f"Iniciando simplificação do texto: {text['id']} [{processed_texts}/{total_texts}]")

        start_time = time.perf_counter()
        state = LinguagemSimplesGraph.run(text['juridico'], model_name=args.model)
        end_time = time.perf_counter()

        time_elapsed = end_time - start_time
        total_time_elapsed += time_elapsed
        total_input_tokens += state['input_tokens']
        total_output_tokens += state['output_tokens']
        llm_calls += state['llm_calls']
        processed_texts += 1

        print("Finalizado!")
        print(f"\tTempo de execução: {time_elapsed:.4f}s")
        print(f"\tTokens: input={state['input_tokens']}, output={state['output_tokens']}")
        print("f\tSalvando resultados para o CSV")
        with open(f"output/{args.model}/metadata_per_text.csv", "a") as f:
            f.write(f"{time_elapsed:.4f},{state['input_tokens']},{state['output_tokens']},{state['llm_calls']}\n")

    print("---------------------")
    print("Dados gerais:")
    print(f"\ttempo={total_time_elapsed:.4f}s")
    print(f"\ttokens: input={total_input_tokens}, output={total_output_tokens}")
    print(f"\tchamadas ao modelo: {llm_calls}")

    with open(f"output/{args.model}/metadata_general.csv", "w") as f:
        f.write("elapsed_time,input_tokens,output_tokens,llm_calls\n")
        f.write(f"{total_time_elapsed:.4f},{total_input_tokens},{total_output_tokens},{llm_calls}\n")
