import random
import asyncio
from http import HTTPStatus
import re
import dashscope
from .common import CommonTranslator
from typing import List, Dict

class QwenBaseTranslator(CommonTranslator):
    _LANGUAGE_CODE_MAP = {
        'CHS': 'Simplified Chinese',
        'CHT': 'Traditional Chinese',
        'CSY': 'Czech',
        'NLD': 'Dutch',
        'ENG': 'English',
        'FRA': 'French',
        'DEU': 'German',
        'HUN': 'Hungarian',
        'ITA': 'Italian',
        'JPN': 'Japanese',
        'KOR': 'Korean',
        'PLK': 'Polish',
        'PTB': 'Portuguese',
        'ROM': 'Romanian',
        'RUS': 'Russian',
        'ESP': 'Spanish',
        'TRK': 'Turkish',
        'UKR': 'Ukrainian',
        'VIN': 'Vietnamese',
        'CNR': 'Montenegrin',
        'SRP': 'Serbian',
        'HRV': 'Croatian',
        'ARA': 'Arabic',
        'THA': 'Thai',
        'IND': 'Indonesian'
    }
    _MODEL_NAME = ''
    _CONFIG_KEY = ''
    _MAX_TOKENS = 32768
    _TIMEOUT = 420  # 7分钟的超时时间
    _RETRY_ATTEMPTS = 3  # 最大重试次数
    _RETURN_PROMPT = False
    _INCLUDE_TEMPLATE = True
    _PROMPT_TEMPLATE = 'Please help me to translate the following text from a manga to {to_lang} (if it\'s already in {to_lang} or looks like gibberish you have to output it as it is instead):\n'

    def __init__(self):
        super().__init__()
        # 初始化配置

    def parse_args(self, args):
        self.config = args.gpt_config
        
    def _config_get(self, key: str, default=None):
        if not self.config:
            return default
        return self.config.get(self._CONFIG_KEY + '.' + key, self.config.get(key, default))

    @property
    def prompt_template(self) -> str:
        return self._config_get('prompt_template', default=self._PROMPT_TEMPLATE)
    
    @property
    def temperature(self) -> float:
        return self._config_get('temperature', default=0.5)
    
    @property
    def top_p(self) -> float:
        return self._config_get('top_p', default=1)
    
    @property
    def chat_system_template(self) -> str:
        return self._config_get('chat_system_template', 'You are a helpful assistant.')

    @property
    def chat_sample(self) -> Dict[str, List[str]]:
        return self._config_get('chat_sample', {})

    # 组装messages，由system user assistant三部分组成
    def _assemble_prompts(self, to_lang: str, queries: List[str]) -> List[Dict]:
        messages = []

        # 添加 system 部分
        system_message = {
            'role': 'system',
            'content': self.chat_system_template.format(to_lang=to_lang)
        }
        messages.append(system_message)

        # 如果有 chat_sample，则按顺序插入 user 和 assistant 部分
        if to_lang in self.chat_sample:
            messages.append({'role': 'user', 'content': self.chat_sample[to_lang][0]})
            messages.append({'role': 'assistant', 'content': self.chat_sample[to_lang][1]})

        # 构造 user messages，仿照 ChatGPT 的逻辑
        if self._INCLUDE_TEMPLATE:
            prompt = self.prompt_template.format(to_lang=to_lang)
        else:
            prompt = ''

        if self._RETURN_PROMPT:
            prompt += '\nOriginal:'

        i_offset = 0
        for i, query in enumerate(queries):
            prompt += f'\n<|{i+1-i_offset}|>{query}'

            # 分割长查询
            if self._MAX_TOKENS * 2 and len(''.join(queries[i+1:])) > self._MAX_TOKENS:
                if self._RETURN_PROMPT:
                    prompt += '\n<|1|>'
                # 将当前构造的 prompt 作为 user message 插入
                messages.append({'role': 'user', 'content': prompt.lstrip()})
                # 重置 prompt 并重新开始计数
                prompt = self.prompt_template.format(to_lang=to_lang) if self._INCLUDE_TEMPLATE else ''
                i_offset = i + 1

        if prompt:  # 确保最后的 prompt 也被添加
            if self._RETURN_PROMPT:
                prompt += '\n<|1|>'
            messages.append({'role': 'user', 'content': prompt.lstrip()})

        return messages
    
    # 执行请求
    async def _perform_request(self, to_lang: str, prompt: str) -> str:
        retry_attempt = 0
        while retry_attempt < self._RETRY_ATTEMPTS:
            try:
                response = dashscope.Generation.call(
                    self._MODEL_NAME,
                    messages=prompt,
                    seed=random.randint(1, 10000),
                    result_format='message',
                )
                self.logger.debug(f'★千问返回: {response}')
                if response.status_code == HTTPStatus.OK:
                    return response.output['choices'][0]['message']['content']
                else:
                    return ''
                    # self.logger.warn(f'Request failed with status {response.status_code}, retrying... Attempt: {retry_attempt + 1}')
            except asyncio.TimeoutError as e:
                self.logger.warn(f'Request timeout, retrying... Attempt: {retry_attempt + 1}')
            except Exception as e:
                self.logger.error(f'An error occurred: {str(e)}, retrying... Attempt: {retry_attempt + 1}')
            retry_attempt += 1
            await asyncio.sleep(1)  # 重试前简单等待1秒

        # 如果所有重试尝试都失败了，抛出异常
        raise Exception(f'Failed to translate after {self._RETRY_ATTEMPTS} attempts')
    
    # 处理响应
    def _process_response(self, response: str, query_size: int) -> List[str]:
        new_translations = re.split(r'<\|\d+\|>', response)
        # 如果第一个元素为空，则移除它
        if not new_translations[0].strip():
            new_translations = new_translations[1:]
        
        # 如果分割后的翻译结果数量小于查询数量，并且查询数量大于1，尝试使用换行符分割
        if len(new_translations) <= 1 and query_size > 1:
            new_translations = re.split(r'\n', response)
        
        # 如果分割后的结果数量不等于查询数量，则返回一个空列表
        if len(new_translations) != query_size:
            return []
        
        return [t.strip() for t in new_translations]
    
        # 重写翻译方法
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        translations = []
        messages = self._assemble_prompts(to_lang, queries)
        self.logger.debug(f'★构造提示词: {messages}')
        
        # 使用self._MODEL_NAME属性来确定模型
        response = await self._perform_request(to_lang, messages)
        
        # 处理翻译结果
        new_translations = self._process_response(response, len(queries))
        translations.extend(new_translations)
        self.logger.debug(f'★翻译结果: {translations}')
        return translations
    
class QwenOfficialTurboTranslator(QwenBaseTranslator):
    _CONFIG_KEY = 'qwen_turbo'
    _MODEL_NAME = dashscope.Generation.Models.qwen_turbo
    
class QwenOfficialPlusTranslator(QwenBaseTranslator):
    _CONFIG_KEY = 'qwen_plus'
    _MODEL_NAME = dashscope.Generation.Models.qwen_plus