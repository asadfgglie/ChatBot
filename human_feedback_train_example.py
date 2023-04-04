from transformers import BertTokenizerFast, AutoModelForCausalLM, GPT2LMHeadModel
from torch.optim import AdamW
import torch

model_name = "tmp\\test_model"

# 載入模型及tokenizer
tokenizer:BertTokenizerFast = BertTokenizerFast.from_pretrained(model_name)
model:GPT2LMHeadModel = AutoModelForCausalLM.from_pretrained(model_name)

# 定義參數
learning_rate = 5e-5
num_epochs = 3
batch_size = 1

# 訓練資料
train_dataset = [
    ["你好嗎?", "我很好，謝謝你。", "你有什麼興趣?", "我喜歡閱讀和旅行。"],
    ["今天天氣真好!", "是啊，真的很適合出門走走。"]
]

# 定義訓練迴圈
optimizer = AdamW(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i in range(len(train_dataset)):
        input_text = ''
        input_id = target_id = attention_mask = None
        for index in range(len(train_dataset[i]) - 1):
            if index != 0:
                input_text += tokenizer.sep_token
            input_text += train_dataset[i][index]
            target_text = train_dataset[i][index + 1]
            input_ids = tokenizer(input_text, return_tensors="pt", padding='max_length')
            target_ids = tokenizer.encode(input_text + tokenizer.sep_token + target_text, 
                                          return_tensors="pt", padding='max_length')
            target_ids[target_ids == 0] = -100

            if input_id is None:
                input_id: torch.Tensor = input_ids['input_ids']
                target_id: torch.Tensor = target_ids
                attention_mask: torch.Tensor = input_ids['attention_mask']
            else:
                input_id = torch.cat((input_id, input_ids['input_ids']))
                target_id = torch.cat((target_id, target_ids))
                attention_mask = torch.cat((attention_mask, input_ids['attention_mask']))

        loss:torch.Tensor = model(input_ids=input_id, 
                                    labels=target_id, 
                                    attention_mask=attention_mask).loss
        loss.backward()
        optimizer.step()
        model.zero_grad()

    print("Epoch:", epoch+1, "Loss:", loss.item())
    
    # 人類反饋學習
    full_break = False
    new_data = []

    new_done = False
    while True:
        input_text = input('使用者輸入')
        print(input_text, flush=True)
        new_data.append(input_text)
        
        input_ids = tokenizer.encode(tokenizer.sep_token.join(new_data), return_tensors="pt")

        re = True
        while re:
            re = False
            output = model.generate(input_ids=input_ids, generation_config=model.generation_config, num_return_sequences=3)
            
            response = [''.join(tokenizer.decode(output[i][input_ids.shape[-1]:], skip_special_tokens=True).split()) for i in range(3)]
            for i in range(3):
                print(f"{i}：", response[i], flush=True)
            print(flush=True)

            while True:
                feedback = input("選擇喜歡的回答(number/re/stop)")
                try:
                    new_data.append(response[int(feedback)])
                    print('chose:', response[int(feedback)], flush=True)
                    break
                except:
                    if feedback == 're':
                        re = True
                        break
                    elif feedback == 'stop':
                        train_dataset.append(new_data)
                        full_break = True
                        break
                    elif feedback == 'give':
                        new_data.append(input('人類提供的答案'))
                        print(new_data[-1], flush=True)
                        break
        if full_break:
            break
