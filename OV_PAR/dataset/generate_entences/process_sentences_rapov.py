multi_templates = ['The person is {a {Gender}|}{ {Role}|}{ with {Age}|}{ and {Body} build|}, {, wearing {UpperBody} on the upper body|}{, which is {UpperClothingColor}|}, {, and {LowerBody} on the lower body|}{, which is {LowerClothingColor}|}, {, with {Shoes}|}{, which are {ShoesColor}|}, {, styled with {Head}|}{, carrying {Attach}|}{, currently {Action}|}, {, {Occlusionwhere}|}{, {Occlusionby}|}.']
import re
group_templates = {'Gender': 'The person is {Gender}.', 'Age': "The person's age is {Age}.", 'Body': 'The person has a {Body} body shape.', 'Role': 'The person is a {Role}.', 'Head': 'The person has {Head}.', 'UpperBody': 'The person is wearing {UpperBody} on the upper body.', 'LowerBody': 'The person is wearing {LowerBody} on the lower body.', 'Shoes': 'The person is wearing {Shoes}.', 'Attach': 'The person is carrying {Attach}.', 'Action': 'The person is {Action}.', 'UpperClothingColor': 'The upper clothing is {UpperClothingColor}.', 'LowerClothingColor': 'The lower clothing is {LowerClothingColor}.', 'ShoesColor': 'The shoes are {ShoesColor}.', 'Occlusionwhere': 'The person is occluded {Occlusionwhere}.', 'Occlusionby': 'The person is occluded by {Occlusionby}.'}
group_absence_templates = {'Gender': ['The gender is unknown.'], 'Age': ['The age of the person is unknown.'], 'Body': ['The body shape of the person is unknown.'], 'Role': ['The role of this person is unknown.'], 'Head': ['No distinctive hairstyle or accessories visible.'], 'UpperBody': ['The top wear of the person is unknown.'], 'LowerBody': ['The lower wear of the person is unknown.'], 'Shoes': ['Footwear is not clearly visible.'], 'Attach': ['This person is not carrying anything.'], 'Action': ['The person is not doing any identifiable action.'], 'UpperClothingColor': ['Upper clothing color is not visible.'], 'LowerClothingColor': ['Lower clothing color is not visible.'], 'ShoesColor': ['Shoe color is not visible.'], 'Occlusionwhere': ['No information on occlusion position.'], 'Occlusionby': ['No information on occlusion cause.']}

def clean_template(template, attrs):
    template = re.sub('\\{[^{}]*?\\|[^{}]*?\\}', '', template)
    template = re.sub('\\s+', ' ', template)
    return re.sub('(\\s*\\,\\s*)*\\.$', '.', template)
multi_attr_words = {'female': ['female'], 'male': ['male'], 'age less 16': ['age less 16 years old'], 'age 17 30': ['age 17 to 30 years old'], 'age 31 45': ['age 31 to 45 years old'], 'age 46 60': ['age 46 to 60 years old'], 'body fat': ['fat'], 'body normal': ['normal'], 'body thin': ['thin'], 'customer': ['customer'], 'employee': ['employee'], 'head bald head': ['bald head'], 'head long hair': ['long hair'], 'head black hair': ['black hair'], 'head hat': ['hat'], 'head glasses': ['glasses'], 'upper shirt': ['shirt'], 'upper sweater': ['sweater'], 'upper vest': ['vest'], 'upper t-shirt': ['t-shirt'], 'upper cotton': ['cotton coat'], 'upper jacket': ['jacket'], 'upper suit up': ['suit'], 'upper tight': ['tight top'], 'upper short sleeve': ['short sleeve top'], 'upper other': ['other upper clothing'], 'lower long trousers': ['long trousers'], 'lower skirt': ['skirt'], 'lower short skirt': ['short skirt'], 'lower dress': ['dress'], 'lower jeans': ['jeans'], 'lower tight trousers': ['tight trousers'], 'shoes leather': ['leather shoes'], 'shoes sport': ['sport shoes'], 'shoes boots': ['boots'], 'shoes cloth': ['cloth shoes'], 'shoes casual': ['casual shoes'], 'shoes other': ['other shoes'], 'attach backpack': ['backpack'], 'attach shoulder bag': ['shoulder bag'], 'attach hand bag': ['hand bag'], 'attach box': ['box'], 'attach plastic bag': ['plastic bag'], 'attach paper bag': ['paper bag'], 'attach hand trunk': ['hand trunk'], 'attach other': ['other items'], 'action calling': ['making a phone call'], 'action talking': ['talking'], 'action gathering': ['gathering something'], 'action holding': ['holding something'], 'action pushing': ['pushing something'], 'action pulling': ['pulling something'], 'action carry arm': ['carrying something on the arm'], 'action carry hand': ['carrying something by hand'], 'action other': ['doing other actions'], 'upper color black': ['black upper clothing'], 'upper color white': ['white upper clothing'], 'upper color gray': ['gray upper clothing'], 'upper color red': ['red upper clothing'], 'upper color green': ['green upper clothing'], 'upper color blue': ['blue upper clothing'], 'upper color silver': ['silver upper clothing'], 'upper color yellow': ['yellow upper clothing'], 'upper color brown': ['brown upper clothing'], 'upper color purple': ['purple upper clothing'], 'upper color pink': ['pink upper clothing'], 'upper color orange': ['orange upper clothing'], 'lower color black': ['black lower clothing'], 'lower color white': ['white lower clothing'], 'lower color gray': ['gray lower clothing'], 'lower color red': ['red lower clothing'], 'lower color green': ['green lower clothing'], 'lower color blue': ['blue lower clothing'], 'lower color silver': ['silver lower clothing'], 'lower color yellow': ['yellow lower clothing'], 'lower color brown': ['brown lower clothing'], 'lower color purple': ['purple lower clothing'], 'lower color pink': ['pink lower clothing'], 'lower color orange': ['orange lower clothing'], 'shoes color black': ['black shoes'], 'shoes color white': ['white shoes'], 'shoes color gray': ['gray shoes'], 'shoes color red': ['red shoes'], 'shoes color green': ['green shoes'], 'shoes color blue': ['blue shoes'], 'shoes color silver': ['silver shoes'], 'shoes color yellow': ['yellow shoes'], 'shoes color brown': ['brown shoes'], 'shoes color purple': ['purple shoes'], 'shoes color pink': ['pink shoes'], 'shoes color orange': ['orange shoes'], 'occlusion left': ['occluded on the left'], 'occlusion right': ['occluded on the right'], 'occlusion above': ['occluded from above'], 'occlusion below': ['occluded from below'], 'occlusion environment': ['occluded by the environment'], 'occlusion attach': ['occluded by an attachment'], 'occlusion person': ['occluded by another person'], 'occlusion other': ['occluded by other factors']}
attribute_groups = {'Gender': {'index': slice(0, 1), 'words': ['female', 'male']}, 'Age': {'index': slice(1, 5), 'words': ['age less 16', 'age 17 30', 'age 31 45', 'age 46 60']}, 'Body': {'index': slice(5, 8), 'words': ['body fat', 'body normal', 'body thin']}, 'Role': {'index': slice(8, 10), 'words': ['customer', 'employee']}, 'Head': {'index': slice(10, 15), 'words': ['head bald head', 'head long hair', 'head black hair', 'head hat', 'head glasses']}, 'UpperBody': {'index': slice(15, 25), 'words': ['upper shirt', 'upper sweater', 'upper vest', 'upper t-shirt', 'upper cotton', 'upper jacket', 'upper suit up', 'upper tight', 'upper short sleeve', 'upper other']}, 'LowerBody': {'index': slice(25, 31), 'words': ['lower long trousers', 'lower skirt', 'lower short skirt', 'lower dress', 'lower jeans', 'lower tight trousers']}, 'Shoes': {'index': slice(31, 37), 'words': ['shoes leather', 'shoes sport', 'shoes boots', 'shoes cloth', 'shoes casual', 'shoes other']}, 'Attach': {'index': slice(37, 45), 'words': ['attach backpack', 'attach shoulder bag', 'attach hand bag', 'attach box', 'attach plastic bag', 'attach paper bag', 'attach hand trunk', 'attach other']}, 'Action': {'index': slice(45, 54), 'words': ['action calling', 'action talking', 'action gathering', 'action holding', 'action pushing', 'action pulling', 'action carry arm', 'action carry hand', 'action other']}, 'UpperClothingColor': {'index': slice(54, 66), 'words': ['upper color black', 'upper color white', 'upper color gray', 'upper color red', 'upper color green', 'upper color blue', 'upper color silver', 'upper color yellow', 'upper color brown', 'upper color purple', 'upper color pink', 'upper color orange']}, 'LowerClothingColor': {'index': slice(66, 78), 'words': ['lower color black', 'lower color white', 'lower color gray', 'lower color red', 'lower color green', 'lower color blue', 'lower color silver', 'lower color yellow', 'lower color brown', 'lower color purple', 'lower color pink', 'lower color orange']}, 'ShoesColor': {'index': slice(78, 90), 'words': ['shoes color black', 'shoes color white', 'shoes color gray', 'shoes color red', 'shoes color green', 'shoes color blue', 'shoes color silver', 'shoes color yellow', 'shoes color brown', 'shoes color purple', 'shoes color pink', 'shoes color orange']}, 'Occlusionwhere': {'index': slice(90, 94), 'words': ['occlusion left', 'occlusion right', 'occlusion above', 'occlusion below']}, 'Occlusionby': {'index': slice(94, 98), 'words': ['occlusion environment', 'occlusion attach', 'occlusion person', 'occlusion other']}}
absence_templates = {'Gender': ['. The gender of the person is unknown'], 'Age': ['. The age of the person is unknown'], 'Body': ['. The stature of this person is unknown'], 'Role': ['. The role of the person is unknown'], 'Head': ['. No distinctive hairstyle visible'], 'UpperBody': ['. The top wear of the person is unknown'], 'LowerBody': ['. The lower wear of the person is unknown'], 'Shoes': ['. Footwear not clearly visible'], 'Attach': ['. This person is not carrying anything'], 'Action': ['. The person is not doing anything'], 'UpperClothingColor': ['. The color of upper clothing is unknown'], 'LowerClothingColor': ['. The color of lower clothing is unknown'], 'ShoesColor': ['. The color of the shoes is unknown'], 'Occlusionwhere': ['. It is unknown where the person is occluded'], 'Occlusionby': ['. It is unknown what occludes the person']}
miss_pattern = {'Gender': '\\{[^{}]*\\{Gender\\}[^{}]*\\}', 'Head': '\\{[^{}]*\\{Head\\}[^{}]*\\}', 'Attach': '\\{[^{}]*\\{Attach\\}[^{}]*\\}', 'Role': '\\{[^{}]*\\{Role\\}[^{}]*\\}', 'Body': '\\{[^{}]*\\{Body\\}[^{}]*\\}', 'Action': '\\{[^{}]*\\{Action\\}[^{}]*\\}', 'Shoes': '\\{[^{}]*\\{Shoes\\}[^{}]*\\}', 'Age': '\\{[^{}]*\\{Age\\}[^{}]*\\}', 'UpperBody': '\\{[^{}]*\\{UpperBody\\}[^{}]*\\}', 'LowerBody': '\\{[^{}]*\\{LowerBody\\}[^{}]*\\}', 'UpperClothingColor': '\\{[^{}]*\\{UpperClothingColor\\}[^{}]*\\}', 'LowerClothingColor': '\\{[^{}]*\\{LowerClothingColor\\}[^{}]*\\}', 'ShoesColor': '\\{[^{}]*\\{ShoesColor\\}[^{}]*\\}', 'Occlusionwhere': '\\{[^{}]*\\{Occlusionwhere\\}[^{}]*\\}', 'Occlusionby': '\\{[^{}]*\\{Occlusionby\\}[^{}]*\\}'}
import numpy as np
import random
import re
random.seed(0)

def process_template(template, attrs_exists, attr_values):
    pattern = re.compile('\\{([^{}]*?)\\|([^{}]*?)\\}')
    while True:
        match = pattern.search(template)
        if not match:
            break
        full = match.group(0)
        condition = match.group(1).strip()
        alternative = match.group(2).strip()
        required_attrs = re.findall('\\{(\\w+)\\}', condition)
        condition_met = all((attrs_exists.get(a, False) for a in required_attrs))
        template = template.replace(full, condition if condition_met else alternative, 1)
    for attr in ['Age', 'Gender', 'UpperBody', 'LowerBody', 'Shoes', 'Head', 'Attach', 'Action', 'Role', 'Body', 'UpperClothingColor', 'LowerClothingColor', 'ShoesColor', 'Occlusionwhere', 'Occlusionby']:
        if f'{{{attr}}}' in template:
            if attr_values.get(attr, '') == '':
                template = re.sub(miss_pattern[attr], random.choice(absence_templates[attr]), template)
            else:
                template = template.replace(f'{{{attr}}}', attr_values.get(attr, ''))
    return template

def clean_sentence(sentence):
    sentence = re.sub('\\{([^{}]*)\\|\\}', '\\1', sentence)
    sentence = re.sub('[{}]', '', sentence)
    sentence = re.sub(',\\s*,', ', ', sentence)
    sentence = re.sub(',\\s*\\.', '.', sentence)
    sentence = re.sub('\\s* \\.', '.', sentence)
    sentence = re.sub('\\s+', ' ', sentence).strip()
    if not sentence.endswith('.'):
        sentence += '.'
    if sentence.startswith('. '):
        sentence = sentence[2:]
    return sentence

def generate_sentences(labels, group_name=None, mode='base'):
    sentences = []
    group_sentences = []
    for label in labels:
        cur_sentences = []
        cur_group_sentences = []
        attrs_exists = {}
        attr_values = {}
        attribute_groups_new = {}
        if mode == 'base':
            assert labels.shape[-1] == 54
            start_index = 0
            group_names = list(attribute_groups.keys())[:10]
            for group_name in group_names:
                attribute_groups_new[group_name] = attribute_groups[group_name]
        elif mode == 'all':
            assert labels.shape[-1] == 98
            start_index = 0
            attribute_groups_new = attribute_groups
        elif mode == 'novel':
            assert labels.shape[-1] == 44
            start_index = 54
            group_names = list(attribute_groups.keys())[10:]
            for group_name in group_names:
                attribute_groups_new[group_name] = attribute_groups[group_name]
        for group in attribute_groups_new:
            config = attribute_groups[group]
            if group == 'Gender':
                gender_val = label[0]
                gender = 'female' if gender_val == 1 else 'male'
                attrs_exists[group] = True
                attr_values[group] = random.choice(multi_attr_words[gender])
            else:
                slice_obj = config['index']
                group_labels = label[[i - start_index for i in slice_obj]]
                exists = np.any(group_labels == 1)
                attrs_exists[group] = exists
                if exists:
                    if len(np.where(group_labels == 1)[0]) > 1:
                        for iter, active_idx in enumerate(np.where(group_labels == 1)[0]):
                            attr_name = config['words'][active_idx]
                            synonyms = multi_attr_words.get(attr_name, [attr_name])
                            if iter < 1:
                                attr_values[group] = random.choice(synonyms)
                            else:
                                attr_values[group] = attr_values[group] + ' and ' + random.choice(synonyms)
                    else:
                        active_idx = np.where(group_labels == 1)[0][0]
                        attr_name = config['words'][active_idx]
                        synonyms = multi_attr_words.get(attr_name, [attr_name])
                        attr_values[group] = random.choice(synonyms)
                else:
                    attr_values[group] = ''
            group_sentence = process_template(group_templates[group], {group: attrs_exists[group]}, {group: attr_values[group]})
            if attrs_exists[group]:
                cur_group_sentences.append(group_sentence)
            else:
                cur_group_sentences.append(random.choice(group_absence_templates[group]))
        sentence = process_template(multi_templates[0], attrs_exists, attr_values)
        cleaned = clean_sentence(sentence)
        if any(attr_values.values()):
            cur_sentences.append(cleaned)
        sentences.extend(cur_sentences)
        group_sentences.append(cur_group_sentences)
    return (sentences, group_sentences)

def generate_sentences_group(labels, group_name=None):
    group_sentences = []
    for label in labels:
        cur_group_sentences = []
        attrs_exists = {}
        attr_values = {}
        config = attribute_groups[group_name]
        if group_name == 'Gender':
            gender_val = label[0]
            gender = 'female' if gender_val == 1 else 'male'
            attrs_exists[group_name] = True
            attr_values[group_name] = random.choice(multi_attr_words[gender])
        else:
            group_labels = label
            exists = np.any(group_labels == 1)
            attrs_exists[group_name] = exists
            if exists:
                if len(np.where(group_labels == 1)[0]) > 1:
                    for iter, active_idx in enumerate(np.where(group_labels == 1)[0]):
                        attr_name = config['words'][active_idx]
                        synonyms = multi_attr_words.get(attr_name, [attr_name])
                        if iter < 1:
                            attr_values[group_name] = random.choice(synonyms)
                        else:
                            attr_values[group_name] = attr_values[group_name] + ' and ' + random.choice(synonyms)
                else:
                    active_idx = np.where(group_labels == 1)[0][0]
                    attr_name = config['words'][active_idx]
                    synonyms = multi_attr_words.get(attr_name, [attr_name])
                    attr_values[group_name] = random.choice(synonyms)
            else:
                attr_values[group_name] = ''
        group_sentence = process_template(group_templates[group_name], {group_name: attrs_exists[group_name]}, {group_name: attr_values[group_name]})
        if attrs_exists[group_name]:
            cur_group_sentences.append(group_sentence)
        else:
            cur_group_sentences.append(random.choice(group_absence_templates[group_name]))
        group_sentences.append(cur_group_sentences)
    return group_sentences
if __name__ == '__main__':
    dummy_labels = np.array([[1] + [1, 0, 0, 0] + [1, 0, 0] + [1, 0] + [0, 0, 0, 0, 1] + [0, 0, 0, 1, 0, 0, 1, 0, 0, 0] + [0, 0, 0, 0, 0, 1] + [0, 0, 0, 0, 0, 1] + [0, 0, 0, 0, 0, 1, 0, 0] + [0, 0, 0, 0, 0, 1, 0, 0, 1] + [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0] + [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] + [0, 1, 0, 0] + [0, 1, 0, 0]])
    group_sizes = [1, 4, 3, 2, 5, 10, 6, 6, 8, 9, 12, 12, 12, 4, 4]
    print(sum(group_sizes))
    print(dummy_labels.shape)
    generated = generate_sentences(dummy_labels, mode='all')
    for i, sents in enumerate(generated):
        print(f'Sample{i + 1}:')
        for j, sent in enumerate(sents):
            print(f'{j + 1}: {sent}')
