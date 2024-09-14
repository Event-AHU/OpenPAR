import random
import numpy as np
random.seed(42)
np.random.seed(42)

template = "The {年龄} {性别} {身材}{头发},{朝向} wearing {上装} on tops{下装}{鞋子}{附件}{动作}."
attr_words = {
    "性别": ['woman'],
    "年龄": ['child', 'adult', 'elderly'],
    "身材": ['with fat body', 'with normal body', 'with thin body'],
    "头发": [', and has bald head', 
           ', and has long hair', 
           ', and has black hair', 
           ', and with hat', 
           ', and with glasses',
           ', and with mask',
           ', and with helmet',
           ', and with scarf', 
           ", and with gloves"],
    "朝向": [' is facing to front,', ' is facing to side,', ' is facing to back,'],
    "上装": [
            'short sleeves', 
           'long sleeves', 
           'shirt',
           'jacket',
           'suit up',
           'vest', 
           'cotton-padded coat', 
           'coat', 
           'graduation gown',
           'chef uniform'],
    "下装": [', and trousers on bottoms', 
           ', and shorts on bottoms',
           ', and jeans on bottoms', 
           ', and long skirt on bottoms',
           ', and short skirt on bottoms',
           ', and dress on bottoms'],
    "鞋子": [", with leather shoes",
           ", with casual shoes",
           ", with boots",
           ", with sandals",
           ", with other shoes"],
    "附件": [", and carrying backpack",
           ", and carrying shoulder bag", 
           ", and carrying hand bag",
           ", and carrying plastic bags", 
           ", and carrying paper bags",
           ', and carrying suitcase',
           ', and carrying other bag'],
    "动作": [', is calling',
           ', is smoking',
           ', is hands behind back',
           ', is arms crossed', 
           ', is walking',
           ', is running',
           ', is standing',
           ', is riding a bicycle',
           ', is riding a scooter',
           ', is riding a skateboard'],
}

group_num=[len(v) for k,v in attr_words.items()]
attr_words["性别"] = ['woman','man']
print(group_num)
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
                            attributes[category] = attributes[category]+ ' and ' + attr_word.replace(', and with ','').replace(', and has ','').replace(', is ','').replace(', and ','')
                        else:
                            attributes[category] = attr_word
        for category in attr_words.keys():
            if category not in attributes:
                attributes[category] = ''
        sentence = template.format(**attributes)
        sentences.append(sentence)
    return sentences


