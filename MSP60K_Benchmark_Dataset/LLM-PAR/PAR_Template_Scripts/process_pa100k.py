import random
import numpy as np
random.seed(42)
np.random.seed(42)
template = "The {年龄} {性别}{配饰}, is wearing {上装}, {下装}{附件}."
attr_words = {
    "性别": ['woman'],
    "年龄": ['age over 60', 'age 18 to 60', 'age less 18'],
    "朝向": [', is facing to front,', ', is facing to side,', ', is facing to back,'],
    "配饰": [' has hat', ' has sunglasses'],
    "附件": [", and carrying hand bag",", and carrying shoulder bag", ", and carrying backpack",", and hold objects in front"],
    "上装": ['short sleeve', 'long sleeve', 'stride top', 'logo top', 'plaid top', 'splice top'],
    "下装": ['stripe pants', 'pattern pants', 'long coat', 'trousers', 'shorts', 'skirt and dress', 'boots'],
}
group_num=[len(v) for k,v in attr_words.items()]

attr_words["性别"] =  ['woman','man']
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
                if attr_idx == 0:
                    if label[attr_idx]:
                        attributes[category] = cur_attr_list[0]
                    else:
                        attributes[category] = cur_attr_list[1]
                else:
                    if label[attr_idx]:
                        if category in attributes.keys():
                            attributes[category] = attributes[category] + ' and ' + attr_word.replace(', and ','')
                        else:
                            attributes[category] = attr_word
        for category in attr_words.keys():
            if category not in attributes:
                attributes[category] = ''
        sentence = template.format(**attributes)
        sentences.append(sentence)
    return sentences


