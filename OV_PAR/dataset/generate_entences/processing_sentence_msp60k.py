import re
multi_templates = ['The person is {a {{Age} |}{Gender}}{ who is {Body} build|}{ and faces to {Facing}}{, wearing {UpperBody} on the upper body|}{, and {LowerBody} on the lower body|}{, with {Shoes}|}{, styled with {Head}|}{, carrying {Attach}|}{, currently {Action}|}.']
group_templates = {'Head': 'The person styled with {Head}.', 'Body': 'The person is {Body} body.', 'UpperBody': 'The person wears {UpperBody} on upper body.', 'LowerBody': "The person's lower with {LowerBody}.", 'Shoes': 'The person with {Shoes}.', 'Attach': 'The person carrying {Attach}.', 'Age': "The person's {Age}.", 'Gender': 'The person is {Gender}.', 'Action': 'The person is currently {Action}.', 'Facing': 'The person is facing to {Facing}.'}
group_absence_templates = {'Attach': ['This person is not carrying anything.'], 'Head': ['No distinctive hairstyle visible.'], 'Body': ['The stature of this person is unknown.'], 'Shoes': ['Footwear not clearly visible.'], 'Age': ['The age of the person is unknown.'], 'UpperBody': ['The top wear of the person is unknown.'], 'LowerBody': ['The lower wear of the person is unknown.'], 'Action': ['The person is not doing anything.'], 'Facing': ['The person faces to where is unknown.']}
multi_attr_words = {'female': ['female'], 'male': ['male'], 'Child': ['child'], 'Adult': ['adult'], 'Elderly': ['elderly'], 'Fat': ['fat'], 'Normal': ['normal'], 'Thin': ['thin'], 'Bald': ['bald'], 'Long hair': ['long hair'], 'Black hair': ['black hair'], 'Hat': ['hat'], 'Glasses': ['glasses'], 'Mask': ['mask'], 'Helmet': ['helmet'], 'Scarf': ['scarf'], 'Gloves': ['gloves'], 'Front': ['front'], 'Back': ['back'], 'Side': ['side'], 'Short sleeves': ['short sleeves'], 'Long sleeves': ['long sleeves'], 'Shirt': ['shirt'], 'Jacket': ['jacket'], 'suit': ['Suit'], 'vest': ['Vest'], 'Cotton-padded coat': ['cotton-padded coat'], 'Coat': ['coat'], 'Graduation gown': ['graduation gown'], 'Chef uniform': ['chef uniform'], 'Trousers': ['trousers'], 'Shorts': ['shorts'], 'Jeans': ['jeans'], 'Long skirt': ['long skirt'], 'Short skirt': ['short skirt'], 'Dress': ['dress'], 'Leather shoes': ['leather shoes'], 'Casual shoes': ['casual shoes'], 'Boots': ['boots'], 'Sandals': ['sandals'], 'Other shoes': ['other shoes'], 'Backpack': ['backpack'], 'Shoulder bag': ['shoulder bag'], 'Handbag': ['handbag'], 'Plastic bag': ['plastic bag'], 'Paper bag': ['paper bag'], 'Suitcase': ['suitcase'], 'Others': ['others'], 'Making a phone call': ['making a phone call'], 'Smoking': ['smoking'], 'Hands behind back': ['hands behind back'], 'Arms crossed': ['arms crossed'], 'Walking': ['walking'], 'Running': ['running'], 'Standing': ['standing'], 'Riding a bicycle': ['riding a bicycle'], 'Riding an scooter': ['riding an scooter'], 'Riding a skateboard': ['riding a skateboard']}
attribute_groups = {'Gender': {'index': 0, 'words': ['female', 'male']}, 'Age': {'index': slice(1, 4), 'words': ['Child', 'Adult', 'Elderly']}, 'Body': {'index': slice(4, 7), 'words': ['Fat', 'Normal', 'Thin']}, 'Head': {'index': slice(7, 16), 'words': ['Bald', 'Long hair', 'Black hair', 'Hat', 'Glasses', 'Mask', 'Helmet', 'Scarf', 'Gloves']}, 'Facing': {'index': slice(16, 19), 'words': ['Front', 'Back', 'Side']}, 'UpperBody': {'index': slice(19, 29), 'words': ['Short sleeves', 'Long sleeves', 'Shirt', 'Jacket', 'Suit', 'Vest', 'Cotton-padded coat', 'Coat', 'Graduation gown', 'Chef uniform']}, 'LowerBody': {'index': slice(29, 35), 'words': ['Trousers', 'Shorts', 'Jeans', 'Long skirt', 'Short skirt', 'Dress']}, 'Shoes': {'index': slice(35, 40), 'words': ['Leather shoes', 'Casual shoes', 'Boots', 'Sandals', 'Other shoes']}, 'Attach': {'index': slice(40, 47), 'words': ['Backpack', 'Shoulder bag', 'Handbag', 'Plastic bag', 'Paper bag', 'Suitcase', 'Others']}, 'Action': {'index': slice(47, 57), 'words': ['Making a phone call', 'Smoking', 'Hands behind back', 'Arms crossed', 'Walking', 'Running', 'Standing', 'Riding a bicycle', 'Riding an scooter', 'Riding a skateboard']}}
base_attribute_groups = {'Gender': {'index': 0, 'words': ['female', 'male']}, 'Age': {'index': slice(1, 4), 'words': ['Child', 'Adult', 'Elderly']}, 'Body': {'index': slice(4, 7), 'words': ['Fat', 'Normal', 'Thin']}, 'Head': {'index': slice(7, 16), 'words': ['Bald', 'Long hair', 'Black hair', 'Hat', 'Glasses', 'Mask', 'Helmet', 'Scarf', 'Gloves']}, 'Facing': {'index': slice(16, 19), 'words': ['Front', 'Back', 'Side']}, 'UpperBody': {'index': slice(19, 26), 'words': ['Long sleeves', 'Shirt', 'Jacket', 'Suit', 'Cotton-padded coat', 'Coat', 'Chef uniform']}, 'LowerBody': {'index': slice(26, 28), 'words': ['Trousers', 'Long skirt']}, 'Shoes': {'index': slice(28, 31), 'words': ['Leather shoes', 'Boots', 'Other shoes']}, 'Attach': {'index': slice(31, 36), 'words': ['Handbag', 'Plastic bag', 'Paper bag', 'Suitcase', 'Others']}, 'Action': {'index': slice(36, 40), 'words': ['Making a phone call', 'Smoking', 'Hands behind back', 'Arms crossed']}}
novel_attribute_groups = {'UpperBody': {'index': slice(0, 3), 'words': ['Short sleeves', 'Vest', 'Graduation gown']}, 'LowerBody': {'index': slice(3, 7), 'words': ['Shorts', 'Long skirt', 'Short skirt', 'Dress']}, 'Shoes': {'index': slice(7, 9), 'words': ['Casual shoes', 'Sandals']}, 'Attach': {'index': slice(9, 11), 'words': ['Backpack', 'Shoulder bag']}, 'Action': {'index': slice(11, 17), 'words': ['Walking', 'Running', 'Standing', 'Riding a bicycle', 'Riding an scooter', 'Riding a skateboard']}}
absence_templates = {'Attach': ['. This person is not carrying anything'], 'Head': ['. No distinctive hairstyle visible'], 'Body': ['. The stature of this person is unknown'], 'Shoes': ['. Footwear not clearly visible'], 'Age': ['. The age of the person is unknown'], 'UpperBody': ['. The top wear of the person is unknown'], 'LowerBody': ['. The lower wear of the person is unknown'], 'Action': ['. The person is not doing anything'], 'Facing': ['. The person faces to where is unknown']}
miss_pattern = {'Head': '\\{[^{}]*\\{Head\\}[^{}]*\\}', 'Attach': '\\{[^{}]*\\{Attach\\}[^{}]*\\}', 'Body': '\\{[^{}]*\\{Body\\}[^{}]*\\}', 'Action': '\\{[^{}]*\\{Action\\}[^{}]*\\}', 'Shoes': '\\{[^{}]*\\{Shoes\\}[^{}]*\\}', 'Age': '\\{[^{}]*\\{Age\\}[^{}]*\\}', 'UpperBody': '\\{[^{}]*\\{UpperBody\\}[^{}]*\\}', 'LowerBody': '\\{[^{}]*\\{LowerBody\\}[^{}]*\\}', 'Facing': '\\{[^{}]*\\{Facing\\}[^{}]*\\}'}
import numpy as np
import random
import re
random.seed(0)

def clean_template(template, attrs):
    template = re.sub('\\{[^{}]*?\\|[^{}]*?\\}', '', template)
    template = re.sub('\\s+', ' ', template)
    return re.sub('(\\s*\\,\\s*)*\\.$', '.', template)

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
    for attr in ['Age', 'Gender', 'UpperBody', 'LowerBody', 'Shoes', 'Head', 'Attach', 'Action', 'Body', 'Facing']:
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
        for group in base_attribute_groups:
            config = base_attribute_groups[group]
            if group == 'Gender':
                gender_val = label[0]
                gender = 'female' if gender_val == 1 else 'male'
                attrs_exists[group] = True
                attr_values[group] = random.choice(multi_attr_words[gender])
            else:
                slice_obj = config['index']
                group_labels = label[slice_obj]
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

def generate_sentences_group(labels, group, set_attribute_groups):
    group_sentences = []
    for label in labels:
        cur_group_sentences = []
        attrs_exists = {}
        attr_values = {}
        config = set_attribute_groups[group]
        if group == 'Gender':
            gender_val = label[0]
            gender = 'female' if gender_val == 1 else 'male'
            attrs_exists[group] = True
            attr_values[group] = random.choice(multi_attr_words[gender])
        else:
            group_labels = label
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
        if group == 'Gender':
            gender_val = label[0]
            gender = 'female' if gender_val == 1 else 'male'
            attrs_exists[group] = True
            attr_values[group] = random.choice(multi_attr_words[gender])
        else:
            group_labels = label
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
        group_sentences.append(cur_group_sentences)
    return group_sentences
if __name__ == '__main__':
    dummy_labels = np.array([[1] + [1, 0, 0] + [1, 0, 0] + [0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 1, 0] + [0, 0, 0, 1, 0, 0, 1, 0, 0, 1] + [0, 0, 0, 0, 1, 1] + [0, 0, 0, 1, 1] + [0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]])
    generated = generate_sentences(dummy_labels)
    for i, sents in enumerate(generated):
        print(f'Sample{i + 1}:')
        for j, sent in enumerate(sents):
            print(f'{j + 1}: {sent}')
