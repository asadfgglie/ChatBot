from flask import Flask, request
from transformers import AutoModelForCausalLM, BertTokenizerFast, GPT2LMHeadModel
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('model_path', type=str)
parser.add_argument('port', type=int)

args = parser.parse_args()

model: GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(args.model_path)
tokenizer = BertTokenizerFast.from_pretrained(args.model_path)

config = model.generation_config

app = Flask(__name__)


@app.route("/", methods=["POST"])
def index():

    user_inputs = tokenizer(tokenizer.sep_token.join(request.json['text']),
                            truncation=True,
                            return_tensors='pt',
                            max_length=1024-config.max_new_tokens)

    try:
        out = model.generate(generation_config=config, **user_inputs)
    except Exception as e:
        return str(e), 500

    reply = ''.join(tokenizer.decode(out[0][user_inputs['input_ids'].shape[-1]:], skip_special_tokens=True).split())

    return {'reply': reply}


if __name__ == '__main__':
    app.run('localhost', port=args.port)
