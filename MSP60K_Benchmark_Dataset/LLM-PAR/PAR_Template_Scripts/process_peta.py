import random
import numpy as np
random.seed(42)
np.random.seed(42)

template = "The {年龄} {性别} has {头发} and is wearing {上装}, {下装}, with {鞋子}{附件}."
attr_words = {
    "头发": ['hat', 'muffler', 'head nothing', 'sunglasses', 'long hair'],
    "上装": [
        'casual top', 'formal top', 'jacket', 'logo top', 'plaid top', 
        'short sleeve', 'thin stripes', 'Tshirt', 'other top', 'Vneck'
    ],
    "下装": [
        'casual pants', 'formal pants', 'jeans', 'shorts', ' short skirt', 'trousers'
    ],
    "鞋子": ["leather","sandals","other shoes","sneakers"],
    "附件": [", and carrying backpack",", and carrying other bag", ", and carrying messenger bag",", and carrying without bag", ", and carrying plastic bags",],
    "年龄": ['age less 30', 'age 30 to 45', 'age 45 to 60', 'age over 60'],
    "性别": ['female', 'male']
}
group_num=[5,10,6,4,5,4,1]
def generate_sentence(labels):
    sentences = []
    for lidx, label in enumerate(labels):
        attributes = {}
        for g_idx in range(len(group_num)):
            for attr_idx in range(0+sum(group_num[:g_idx]),0+sum(group_num[:g_idx+1])):
                relative_idx = attr_idx - sum(group_num[:g_idx])
                category = list(attr_words.keys())[g_idx]
                cur_attr_list=attr_words[category]
                attr_word = cur_attr_list[relative_idx]
                if attr_idx == 34:
                    if label[attr_idx]:
                        attributes[category] = cur_attr_list[1]
                    else:
                        attributes[category] = cur_attr_list[0]
                else:
                    if label[attr_idx]:
                        if category in attributes.keys():
                            attributes[category] = attributes[category] + ' and ' + attr_word
                        else:
                            attributes[category] = attr_word
        for category in attr_words.keys():
            if category not in attributes:
                attributes[category] = ''
        sentence = template.format(**attributes)
        sentences.append(sentence)
    return sentences


