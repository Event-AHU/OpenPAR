import random
import numpy as np
random.seed(42)
np.random.seed(42)

template = "The {年龄} {性别} with {身材}, {身份} {头发}, wearing {上装} tops, and {下装} bottoms, with {鞋子}{附件}{动作}."
attr_words = {
    "头发": ['has bald head', 'has long hair', 'has black hair', 'has hat', 'has glasses'],
    "上装": ['shirt', 'sweater', 'vest', 'Tshirt', 'cotton top','jacket', 'suit up', 'tight top', 'short sleeve','other top'],
    "下装": ['long trousers', 'skirt', 'short skirt', 'dress', 'jeans', 'tight trousers'],
    "鞋子": ["leather","sport shoes","boots","cloth shoes","casual shoes","other shoes"],
    "附件": [", and carrying backpack ",", and carrying shoulder bag ", ", and carrying hand bag ",", and carrying box ", 
           ", and carrying plastic bags ",', and carrying paper bag ',', and carrying hand trunk ',', and carrying other bag '],
    "性别": ['woman'],
    "年龄": ['age less 16', 'age 17 to 30', 'age 31 to 45'],
    "身材": ['fat', 'normal body', 'thin'],
    "身份": ['is a customer', 'is a employee'],
    "动作": [', is calling', ', is talking',', is gathering',', is holding', ', is pushing', ', is pulling', ', is carry arm', ', is carry hand', ', is doing somethings'],
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
                if attr_idx == 38:
                    if label[attr_idx]:
                        attributes[category] = cur_attr_list[0]
                    else:
                        attributes[category] = cur_attr_list[1]
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


