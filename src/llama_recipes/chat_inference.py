import os
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
os.environ['HUGGING_FACE_HUB_TOKEN'] = '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'
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n"


model_path = "./ckpt/llama-13b/1102/Ep9" #hyunseoki/ko-en-llama2-13b
# model_path = "hyunseoki/ko-en-llama2-13b"
base_model = AutoModelForCausalLM.from_pretrained("hyunseoki/ko-en-llama2-13b")
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained("hyunseoki/ko-en-llama2-13b", use_fast=True)

hub_name = "kiyoonyoo/1102-volcano"
tokenizer.push_to_hub(hub_name, private=True)
model = model.merge_and_unload()
model.push_to_hub(hub_name, private=True)
model.cuda()
##
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = [], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids, scores):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if (stop == input_ids[0][-len(stop):]).all():
                return True
        return False

stop_words = ["[user] ", " [user] ", "[user]"]
stop_words = ["[user] ", " [user] ", "[user]"]
stop_words = []
stop_words_ids = [tokenizer(stop_word, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze() for stop_word in stop_words]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])


##
text = """
[정보] 
0) 유저 : 조걸은 어떻게 생겼는가?
청명 : 조걸은 잘 생겼어.
1 )유저 : 백천 잘 싸움?
청명 : 나한테 발렸지 크크
2) 유저 : 스쿼트가 뭔지 설명해줘.
청명 : 스쿼트는 무게를 들고 하체에 자극을 주는 운동이야.
3) 유저 : 이대 제자들이 너를 어떻게 봤을까?
청명 : 그런 건 뭐 상관없지. 난 그저 모두를 내 발 아래 두고 굴릴 수 있는 걸 원했을 뿐이야! 으하하하하핫!
---
[대화 기록]
{history}

[질문]
{query}
---
'청명'이 되어 유저의 [질문]에 대해 적절하게 대답합니다.
위의 [정보]를 참고하시오. [정보]가 유저의 [질문]과 관련이 없다면 무시하시오!!
"""
text += "청명 :"
# query = "유저: 스쿼트하고 있다. 넌 헬스 중 어떤 운동해봤어?"
history = "유저: 야 뭐하냐? 나는 헬스 중이다.\n청명: 난 수련 중이다. 헬스에서는 무슨 운동을 하고 있냐?"
query = "유저: 스쿼트하고 있어. 근데 백천 잘 싸워??"
text = text.format(history=history, query=query)
# text = text.format(context=context)
sep_token = "유저 :"
input_ids = tokenizer(text, return_tensors='pt')['input_ids'].cuda()


while True:
    output = model.generate(input_ids, max_new_tokens=200, do_sample=False)
    output_text = tokenizer.decode(output[0])
    output = output[:, input_ids.shape[-1]:]
    bot_response = tokenizer.decode(output[0])
    # bot_response = tokenizer.decode(output[0]).split(sep_token)[1]
    print(bot_response)
    breakpoint()
    print("유저: ", end="")
    user_input = input() + f" {sep_token}"
    output_text += user_input
    input_ids = tokenizer(output_text, return_tensors='pt')['input_ids'].cuda()

