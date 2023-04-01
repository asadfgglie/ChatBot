import time
import discord

from envparse import env

from transformers import AutoModelForCausalLM, BertTokenizerFast, PreTrainedModel

import os

class Bot(discord.Client):
    def __init__(self, loop=None, **options):
        super().__init__(loop=loop, **options)
        self.model_name = 'model/checkpoint-5005254-save'
        self.is_public = False

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()
        self.chat_history = []
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)

        try:
            with open('admin', 'r', encoding='UTF-8') as admin:
                self.admin_id = admin.readline()[:-1]
                self.admin_name = admin.readline()
        except FileNotFoundError:
            pass


    async def on_ready(self):
        print('目前登入身份：', self.user)
        # discord.Status.<狀態>，可以是online,offline,idle,dnd,invisible
        await self.change_presence(status=discord.Status.idle, activity=discord.Game('你真的是太無情啦!GG人'))


    # 當有訊息時
    async def on_message(self, message: discord.message.Message):
        # 排除自己的訊息，避免陷入無限循環
        if message.author == self.user:
            return
        if message.content == '/clear':
            self.chat_history = []
            return
        print(f'{message.author.name}:{message.content}')

        self.chat_history.append(message.content)

        user_input = self.tokenizer(message.content, return_tensors='pt', truncation=True, max_length=1024-self.model.generation_config.max_new_tokens)

        reply = self.model.generate(generation_config=self.model.generation_config, **user_input)
        reply = self.tokenizer.decode(reply[0][user_input['input_ids'].shape[-1]:], skip_special_tokens=True)
        reply = ''.join(reply.split())
        self.chat_history.append(reply)

        await message.channel.send(reply)
        print(f'{self.model_name}:{reply}\n')

        log_name = time.strftime('%Y-%m-%d', time.localtime()) + '.log'

        os.makedirs('./log', exist_ok=True)

        try:
            with open('./log/' + log_name, 'a', encoding='UTF-8') as log:
                log.write(f'{message.author}:{message.content}\n')
                log.write(f'{self.model_name}:{reply}\n')
        except FileNotFoundError:
            with open('./log/' + log_name, 'w', encoding='UTF-8') as log:
                log.write(f'{message.author}:{message.content}\n')
                log.write(f'{self.model_name}:{reply}\n')




env.read_envfile('./dev.env')
intents = discord.Intents.default()
intents.message_content = True
client = Bot(intents=intents)
client.run(os.getenv('BOT_TOKEN'))