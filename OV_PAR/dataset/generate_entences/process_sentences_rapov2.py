multi_templates = ['The person is {a {IdentityTraits}|}, {, wearing {UpperBody} on the upper body|}{, which is {UpperClothingColor}|}, {, and {LowerBody} on the lower body|}{, which is {LowerClothingColor}|}, {, with {Shoes}|}{, which are {ShoesColor}|}, {, styled with {Head}|}{, carrying {Attach}|}{, currently {Action}|}, {, {Occlusion}|}.']
import re
group_templates = {'IdentityTraits': 'The person is {IdentityTraits}.', 'Head': 'The person has {Head}.', 'UpperBody': 'The person is wearing {UpperBody} on the upper body.', 'LowerBody': 'The person is wearing {LowerBody} on the lower body.', 'Shoes': 'The person is wearing {Shoes}.', 'Attach': 'The person is carrying {Attach}.', 'Action': 'The person is {Action}.', 'UpperClothingColor': 'The upper clothing is {UpperClothingColor}.', 'LowerClothingColor': 'The lower clothing is {LowerClothingColor}.', 'ShoesColor': 'The shoes are {ShoesColor}.', 'Occlusion': 'The person is {Occlusion}.'}
group_absence_templates = {'IdentityTraits': ['The gender/age/role/bodyshape is unknown.'], 'Head': ['No distinctive hairstyle or accessories visible.'], 'UpperBody': ['The top wear of the person is unknown.'], 'LowerBody': ['The lower wear of the person is unknown.'], 'Shoes': ['Footwear is not clearly visible.'], 'Attach': ['This person is not carrying anything.'], 'Action': ['The person is not doing any identifiable action.'], 'UpperClothingColor': ['Upper clothing color is not visible.'], 'LowerClothingColor': ['Lower clothing color is not visible.'], 'ShoesColor': ['Shoe color is not visible.'], 'Occlusion': ['No information on occlusion position or case.']}

def clean_template(template, attrs):
    template = re.sub('\\{[^{}]*?\\|[^{}]*?\\}', '', template)
    template = re.sub('\\s+', ' ', template)
    return re.sub('(\\s*\\,\\s*)*\\.$', '.', template)
multi_attr_words = {'female': ['female'], 'male': ['male'], 'age less 16': ['age less 16 years old'], 'age 17 30': ['age 17 to 30 years old'], 'age 31 45': ['age 31 to 45 years old'], 'age 46 60': ['age 46 to 60 years old'], 'body fat': ['fat'], 'body normal': ['normal'], 'body thin': ['thin'], 'customer': ['customer'], 'employee': ['employee'], 'head bald head': ['bald head'], 'head long hair': ['long hair'], 'head black hair': ['black hair'], 'head hat': ['hat'], 'head glasses': ['glasses'], 'upper shirt': ['shirt'], 'upper sweater': ['sweater'], 'upper vest': ['vest'], 'upper t-shirt': ['t-shirt'], 'upper cotton': ['cotton coat'], 'upper jacket': ['jacket'], 'upper suit up': ['suit'], 'upper tight': ['tight top'], 'upper short sleeve': ['short sleeve top'], 'upper other': ['other upper clothing'], 'lower long trousers': ['long trousers'], 'lower skirt': ['skirt'], 'lower short skirt': ['short skirt'], 'lower dress': ['dress'], 'lower jeans': ['jeans'], 'lower tight trousers': ['tight trousers'], 'shoes leather': ['leather shoes'], 'shoes sport': ['sport shoes'], 'shoes boots': ['boots'], 'shoes cloth': ['cloth shoes'], 'shoes casual': ['casual shoes'], 'shoes other': ['other shoes'], 'attach backpack': ['backpack'], 'attach shoulder bag': ['shoulder bag'], 'attach hand bag': ['hand bag'], 'attach box': ['box'], 'attach plastic bag': ['plastic bag'], 'attach paper bag': ['paper bag'], 'attach hand trunk': ['hand trunk'], 'attach other': ['other items'], 'action calling': ['making a phone call'], 'action talking': ['talking'], 'action gathering': ['gathering something'], 'action holding': ['holding something'], 'action pushing': ['pushing something'], 'action pulling': ['pulling something'], 'action carry arm': ['carrying something on the arm'], 'action carry hand': ['carrying something by hand'], 'action other': ['doing other actions'], 'upper color black': ['black upper clothing'], 'upper color white': ['white upper clothing'], 'upper color gray': ['gray upper clothing'], 'upper color red': ['red upper clothing'], 'upper color green': ['green upper clothing'], 'upper color blue': ['blue upper clothing'], 'upper color silver': ['silver upper clothing'], 'upper color yellow': ['yellow upper clothing'], 'upper color brown': ['brown upper clothing'], 'upper color purple': ['purple upper clothing'], 'upper color pink': ['pink upper clothing'], 'upper color orange': ['orange upper clothing'], 'lower color black': ['black lower clothing'], 'lower color white': ['white lower clothing'], 'lower color gray': ['gray lower clothing'], 'lower color red': ['red lower clothing'], 'lower color green': ['green lower clothing'], 'lower color blue': ['blue lower clothing'], 'lower color silver': ['silver lower clothing'], 'lower color yellow': ['yellow lower clothing'], 'lower color brown': ['brown lower clothing'], 'lower color purple': ['purple lower clothing'], 'lower color pink': ['pink lower clothing'], 'lower color orange': ['orange lower clothing'], 'shoes color black': ['black shoes'], 'shoes color white': ['white shoes'], 'shoes color gray': ['gray shoes'], 'shoes color red': ['red shoes'], 'shoes color green': ['green shoes'], 'shoes color blue': ['blue shoes'], 'shoes color silver': ['silver shoes'], 'shoes color yellow': ['yellow shoes'], 'shoes color brown': ['brown shoes'], 'shoes color purple': ['purple shoes'], 'shoes color pink': ['pink shoes'], 'shoes color orange': ['orange shoes'], 'occlusion left': ['occluded on the left'], 'occlusion right': ['occluded on the right'], 'occlusion above': ['occluded from above'], 'occlusion below': ['occluded from below'], 'occlusion environment': ['occluded by the environment'], 'occlusion attach': ['occluded by an attachment'], 'occlusion person': ['occluded by another person'], 'occlusion other': ['occluded by other factors']}
attribute_groups = {'IdentityTraits': {'index': slice(0, 10), 'words': ['female', 'age less 16 years old', 'age 17 to 30 years old', 'age 31 to 45 years old', 'age 46 to 60 years old', 'fat body', 'normal body', 'thin body', 'customer', 'employee']}, 'Head': {'index': slice(10, 15), 'words': ['bald head', 'long hair', 'black hair', 'hat', 'glasses']}, 'UpperBody': {'index': slice(15, 25), 'words': ['shirt', 'sweater', 'vest', 't-shirt', 'cotton coat', 'jacket', 'suit', 'tight top', 'short sleeve top', 'other upper']}, 'LowerBody': {'index': slice(25, 31), 'words': ['long trousers', 'skirt', 'short skirt', 'dress', 'jeans', 'tight trousers']}, 'Shoes': {'index': slice(31, 37), 'words': ['leather shoes', 'sport shoes', 'boots', 'cloth shoes', 'casual shoes', 'other shoes']}, 'Attach': {'index': slice(37, 45), 'words': ['backpack', 'shoulder bag', 'hand bag', 'box', 'plastic bag', 'paper bag', 'hand trunk', 'other items']}, 'Action': {'index': slice(45, 54), 'words': ['making a phone call', 'talking', 'gathering something', 'holding something', 'pushing something', 'pulling something', 'carrying something on the arm', 'carrying something by hand', 'doing other actions']}, 'UpperClothingColor': {'index': slice(54, 66), 'words': ['black upper', 'white upper', 'gray upper', 'red upper', 'green upper', 'blue upper', 'silver upper', 'yellow upper', 'brown upper', 'purple upper', 'pink upper', 'orange upper']}, 'LowerClothingColor': {'index': slice(66, 78), 'words': ['black lower', 'white lower', 'gray lower', 'red lower', 'green lower', 'blue lower', 'silver lower', 'yellow lower', 'brown lower', 'purple lower', 'pink lower', 'orange lower']}, 'ShoesColor': {'index': slice(78, 90), 'words': ['black shoes', 'white shoes', 'gray shoes', 'red shoes', 'green shoes', 'blue shoes', 'silver shoes', 'yellow shoes', 'brown shoes', 'purple shoes', 'pink shoes', 'orange shoes']}, 'Occlusion': {'index': slice(90, 98), 'words': ['occluded on the left', 'occluded on the right', 'occluded from above', 'occluded from below', 'occluded by the environment', 'occluded by an attachment', 'occluded by another person', 'occluded by other factors']}}
base_attribute_groups = {'IdentityTraits': {'index': slice(0, 7), 'words': ['female', 'age 17 to 30 years old', 'age 31 to 45 years old', 'fat body', 'normal body', 'thin body', 'customer']}, 'Head': {'index': slice(7, 10), 'words': ['long hair', 'black hair', 'glasses']}, 'UpperBody': {'index': slice(10, 17), 'words': ['shirt', 'sweater', 'vest', 't-shirt', 'cotton coat', 'jacket', 'short sleeve top']}, 'LowerBody': {'index': slice(17, 21), 'words': ['long trousers', 'skirt', 'jeans', 'tight trousers']}, 'Shoes': {'index': slice(21, 25), 'words': ['leather shoes', 'sport shoes', 'boots', 'casual shoes']}, 'Attach': {'index': slice(25, 30), 'words': ['shoulder bag', 'hand bag', 'box', 'plastic bag', 'other items']}, 'Action': {'index': slice(30, 36), 'words': ['making a phone call', 'talking', 'gathering something', 'holding something', 'carrying something on the arm', 'carrying something by hand']}, 'UpperClothingColor': {'index': slice(36, 44), 'words': ['black upper', 'white upper', 'gray upper', 'red upper', 'green upper', 'blue upper', 'yellow upper', 'pink upper']}, 'LowerClothingColor': {'index': slice(44, 52), 'words': ['black lower', 'white lower', 'gray lower', 'red lower', 'green lower', 'blue lower', 'yellow lower', 'brown lower']}, 'ShoesColor': {'index': slice(52, 60), 'words': ['black shoes', 'white shoes', 'gray shoes', 'red shoes', 'green shoes', 'blue shoes', 'yellow shoes', 'brown lower']}, 'Occlusion': {'index': slice(60, 65), 'words': ['occluded on the left', 'occluded on the above', 'occluded from below', 'occluded by the environment', 'occluded by other factors']}}
novel_attribute_groups = {'IdentityTraits': {'index': slice(0, 3), 'words': ['age less 16 years old', 'age 46 to 60 years old', 'employee']}, 'Head': {'index': slice(3, 5), 'words': ['bald head', 'hat']}, 'UpperBody': {'index': slice(5, 8), 'words': ['suit', 'tight top', 'other upper']}, 'LowerBody': {'index': slice(8, 10), 'words': ['short skirt', 'dress']}, 'Shoes': {'index': slice(10, 12), 'words': ['cloth shoes', 'other shoes']}, 'Attach': {'index': slice(12, 15), 'words': ['backpack', 'paper bag', 'hand trunk']}, 'Action': {'index': slice(15, 18), 'words': ['pushing something', 'pulling something', 'doing other actions']}, 'UpperClothingColor': {'index': slice(18, 22), 'words': ['silver upper', 'brown upper', 'purple upper', 'orange upper']}, 'LowerClothingColor': {'index': slice(22, 26), 'words': ['silver lower', 'purple lower', 'pink lower', 'orange lower']}, 'ShoesColor': {'index': slice(26, 30), 'words': ['silver shoes', 'purple shoes', 'pink shoes', 'orange shoes']}, 'Occlusion': {'index': slice(30, 33), 'words': ['occluded on the right', 'occluded from attachment', 'occluded by an another person']}}
absence_templates = {'IdentityTraits': ['. The gender/age/role/bodyshape of the person is unknown'], 'Head': ['. No distinctive hairstyle visible'], 'UpperBody': ['. The top wear of the person is unknown'], 'LowerBody': ['. The lower wear of the person is unknown'], 'Shoes': ['. Footwear not clearly visible'], 'Attach': ['. This person is not carrying anything'], 'Action': ['. The person is not doing anything'], 'UpperClothingColor': ['. The color of upper clothing is unknown'], 'LowerClothingColor': ['. The color of lower clothing is unknown'], 'ShoesColor': ['. The color of the shoes is unknown'], 'Occlusion': ['. It is unknown where and what of the person is occluded']}
miss_pattern = {'IdentityTraits': '\\{[^{}]*\\{IdentityTraits\\}[^{}]*\\}', 'Head': '\\{[^{}]*\\{Head\\}[^{}]*\\}', 'Attach': '\\{[^{}]*\\{Attach\\}[^{}]*\\}', 'Action': '\\{[^{}]*\\{Action\\}[^{}]*\\}', 'Shoes': '\\{[^{}]*\\{Shoes\\}[^{}]*\\}', 'UpperBody': '\\{[^{}]*\\{UpperBody\\}[^{}]*\\}', 'LowerBody': '\\{[^{}]*\\{LowerBody\\}[^{}]*\\}', 'UpperClothingColor': '\\{[^{}]*\\{UpperClothingColor\\}[^{}]*\\}', 'LowerClothingColor': '\\{[^{}]*\\{LowerClothingColor\\}[^{}]*\\}', 'ShoesColor': '\\{[^{}]*\\{ShoesColor\\}[^{}]*\\}', 'Occlusion': '\\{[^{}]*\\{Occlusion\\}[^{}]*\\}'}
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
    for attr in ['IdentityTraits', 'UpperBody', 'LowerBody', 'Shoes', 'Head', 'Attach', 'Action', 'UpperClothingColor', 'LowerClothingColor', 'ShoesColor', 'Occlusion']:
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

def generate_sentences(labels):
    sentences = []
    group_sentences = []
    for label in labels:
        cur_sentences = []
        cur_group_sentences = []
        attrs_exists = {}
        attr_values = {}
        for group_name in list(attribute_groups.keys()):
            config = attribute_groups[group_name]
            slice_obj = config['index']
            group_labels = label[slice_obj]
            if group_name == 'IdentityTraits':
                gender_val = label[0]
                gender = 'female' if gender_val == 1 else 'male'
                attrs_exists[group_name] = True
                attr_values[group_name] = random.choice(multi_attr_words[gender])
            if True:
                if group_name == 'IdentityTraits':
                    label[0] = 0
                    group_labels = label
                else:
                    group_labels = label
                exists = np.any(group_labels == 1)
                attrs_exists[group_name] = exists
                if exists:
                    iter_start = 1 if group_name == 'IdentityTraits' else 0
                    if len(np.where(group_labels == 1)[0]) > 1 or group_name == 'IdentityTraits':
                        for iter, active_idx in enumerate(np.where(group_labels == 1)[0]):
                            print(active_idx)
                            attr_name = config['words'][active_idx]
                            synonyms = multi_attr_words.get(attr_name, [attr_name])
                            if iter_start + iter < 1:
                                attr_values[group_name] = random.choice(synonyms)
                            elif synonyms[0] in ['employee', 'customer']:
                                attr_values[group_name] = attr_values[group_name] + ' who is a ' + random.choice(synonyms)
                            elif 'age' in synonyms[0] or 'body' in synonyms[0]:
                                attr_values[group_name] = attr_values[group_name] + ' with ' + random.choice(synonyms)
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
        sentence = process_template(multi_templates[0], attrs_exists, attr_values)
        cleaned = clean_sentence(sentence)
        if any(attr_values.values()):
            cur_sentences.append(cleaned)
        sentences.extend(cur_sentences)
        group_sentences.append(cur_group_sentences)
    return (sentences, group_sentences)

def generate_sentences_group(labels, group_name, set_attribute_groups):
    group_sentences = []
    for label in labels:
        cur_group_sentences = []
        attrs_exists = {}
        attr_values = {}
        config = set_attribute_groups[group_name]
        if group_name == 'IdentityTraits':
            gender_val = label[0]
            gender = 'female' if gender_val == 1 else 'male'
            attrs_exists[group_name] = True
            attr_values[group_name] = random.choice(multi_attr_words[gender])
        if True:
            if group_name == 'IdentityTraits':
                label[0] = 0
                group_labels = label
            else:
                group_labels = label
            exists = np.any(group_labels == 1)
            attrs_exists[group_name] = exists
            if exists:
                iter_start = 1 if group_name == 'IdentityTraits' else 0
                if len(np.where(group_labels == 1)[0]) > 1 or group_name == 'IdentityTraits':
                    for iter, active_idx in enumerate(np.where(group_labels == 1)[0]):
                        attr_name = config['words'][active_idx]
                        synonyms = multi_attr_words.get(attr_name, [attr_name])
                        if iter_start + iter < 1:
                            attr_values[group_name] = random.choice(synonyms)
                        elif synonyms[0] in ['employee', 'customer']:
                            attr_values[group_name] = attr_values[group_name] + ' who is a ' + random.choice(synonyms)
                        elif 'age' in synonyms[0] or 'body' in synonyms[0]:
                            attr_values[group_name] = attr_values[group_name] + ' with ' + random.choice(synonyms)
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

def generate_sentences_group2(labels, group, set_type='base'):
    group_sentences = []
    for label in labels:
        cur_group_sentences = []
        attrs_exists = {}
        attr_values = {}
        if set_type == 'base':
            config = base_attribute_groups[group]
        elif set_type == 'novel':
            config = novel_attribute_groups[group]
        elif set_type == 'all':
            config = attribute_groups[group]
        if group == 'IdentityTraits' and set_type != 'novel':
            gender_val = label[0]
            gender = 'female' if gender_val == 1 else 'male'
            attrs_exists[group] = True
            attr_values[group] = random.choice(multi_attr_words[gender])
        if True:
            if group == 'IdentityTraits' and set_type != 'novel':
                label[0] = 0
                group_labels = label
            else:
                group_labels = label
            exists = np.any(group_labels == 1)
            attrs_exists[group] = exists
            if exists:
                if len(np.where(group_labels == 1)[0]) > 1 or (group == 'IdentityTraits' and set_type != 'novel'):
                    iter_start = 1 if group == 'IdentityTraits' else 0
                    iter_start = iter_start if set_type != 'novel' else 0
                    for iter, active_idx in enumerate(np.where(group_labels == 1)[0]):
                        attr_name = config['words'][active_idx]
                        synonyms = multi_attr_words.get(attr_name, [attr_name])
                        if iter + iter_start < 1:
                            attr_values[group] = random.choice(synonyms)
                        elif synonyms[0] in ['employee', 'customer']:
                            attr_values[group] = attr_values[group] + ' who is a ' + random.choice(synonyms)
                        elif 'age' in synonyms[0] or 'body' in synonyms[0]:
                            attr_values[group] = attr_values[group] + ' with ' + random.choice(synonyms)
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
        group_sentences.append(cur_group_sentences)
    return group_sentences
if __name__ == '__main__':
    dummy_labels = np.array([[0, 1, 0, 0, 1, 0, 0] + [0, 0, 0] + [0, 0, 0, 1, 0, 0, 1] + [0, 0, 0, 1] + [0, 1, 0, 0] + [1, 0, 1, 0, 0] + [0, 1, 0, 1, 0, 0] + [0, 0, 0, 0, 0, 1, 0, 0] + [0, 0, 0, 1, 0, 1, 0, 0] + [0, 0, 0, 1, 0, 0, 0, 0] + [0, 0, 0, 1, 0]])
    group_sizes = [3, 2, 3, 2, 2, 3, 3, 4, 4, 4, 3]
    print(sum(group_sizes))
    slices = 0
    print(dummy_labels.shape)
    generated, _ = generate_sentences(dummy_labels)
    for i, sents in enumerate(generated):
        print(f'Sample{i + 1}:')
        for j, sent in enumerate(sents):
            print(f'{j + 1}: {sent}')
