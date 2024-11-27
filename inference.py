"""
sssmmmyyy---PLGG

"""
import time
import torch

from transformers import AutoTokenizer, AutoModel
torch.set_default_tensor_type(torch.cuda.HalfTensor)


def inference(
        model,
        tokenizer,
        instuction: str,
        sentence: str
    ):
    """
    模型 inference 函数。

    Args:
        instuction (str): _description_
        sentence (str): _description_

    Returns:
        _type_: _description_
    """
    with torch.no_grad():
        input_text = f"Instruction: {instuction}\n"
        if sentence:
            input_text += f"Input: {sentence}\n"
        input_text += f"Answer: "
        batch = tokenizer(input_text, return_tensors="pt")
        out = model.generate(
            input_ids=batch["input_ids"].to(device),
            max_new_tokens=max_new_tokens,
            temperature=0
        )
        out_text = tokenizer.decode(out[0])
        answer = out_text.split('Answer: ')[-1]
        return answer


if __name__ == '__main__':
    from rich import print

    device = 'cuda:0'
    max_new_tokens = 300
    model_path = "checkpoints/model_1000"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
    ).half().to(device)

    samples = [
        {
            'instruction': "给你一个句子，同时给你一个句子主语和关系列表，你需要找出句子中主语包含的所有关系以及关系值，并输出为SPO列表。",
            "input": "给定主语：故宫\n给定句子：故宫位于北京市，以三大殿为中心，占地面积约72万平方米，建筑面积约15万平方米，有大小宫殿七十多座。\n给定关系列表：['所在城市', '面积']",
        },
        {
            'instruction': "给你一个句子，同时给你一个句子主语和关系列表，你需要找出句子中主语包含的所有关系以及关系值，并输出为SPO列表。",
            "input": "句子主语：天山天池\n输入句子:天山天池，古称“瑶池”，地处新疆维吾尔自治区昌吉回族自治州阜康市境内，距自治区首府乌鲁木齐市68公里，交通、电讯十分便利。天山天池景区总面积为548平方公里,分8大景区(天池景区、灯杆山景区、马牙山景区、博格达峰景区、白杨沟景区、花儿沟景区、水磨沟景区、北部梧桐沟沙漠景区)，15个景群，38个景点，是我国西北干旱地区典型的山岳型自然景观。\n输入关系:['所在城市', '面积, '著名景点']",
        }
    ]

    start = time.time()
    for i, sample in enumerate(samples):
        res = inference(
            model,
            tokenizer,
            sample['instruction'],
            sample['input']
        )
        print(f'res {i}: ')
        print(res)
    print(f'Used {round(time.time() - start, 2)}s.')