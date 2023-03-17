import pandas as pd
import py_vncorenlp

vn_pos_tag = py_vncorenlp.VnCoreNLP(annotators=["pos"], save_dir='/home/yenvt/minhnt/')

def read_cluster_len():
    data = []
    with open('/home/yenvt/minhnt/trung.txt', 'r') as f:
        for line in f:
            data.append(int(line.strip()))
    return data

def read_file(path, file_name):
    df = pd.read_csv(path + file_name)
    # lines = df['document'][0]
    src_list = []
    for lines in df['document']:
        sent_list = lines.split(". ")
        if sent_list[-1] == '':
            sent_list = sent_list[:-1]
        sent_list = [sent for sent in sent_list if sent != '']
        sent_list = [sent + '.' if sent[-1] != '.' else sent for sent in sent_list ]
        src_list.append(sent_list)
        if len(src_list) == 3: break
    return src_list

def tag_pos(str_text):
    doc = vn_pos_tag.annotate_text(str_text)
    textlist=[]
    for item in doc[0]:
        source_token = item.get('wordForm').lower()
        source_pos = item.get('posTag')
        if source_pos == 'CH':
            source_pos = source_token
        textlist.append(source_token+'/'+source_pos)
    return ' '.join(textlist)

def get_pronoun(doc, tag):
    l = []
    for item in doc:
        if item[1] == tag:
            l.append(item[0].lower())
    return l

def check_pronoun(str1, str2):
    flag = False
    doc1 = get_pronoun(str1, 'P')
    doc2 = get_pronoun(str2, 'P')
    if len(doc1)>0 and len(doc2)>0:
        for text in doc1:
            if text in doc2:
                flag=True
                break
    return flag

def check_noun(str1, str2):
    flag = False
    doc1 = get_pronoun(str1, 'N')
    doc2 = get_pronoun(str2, 'N')
    if len(doc1)>0 and len(doc2)>0:
        for text in doc1:
            if text in doc2:
                flag=True
                break
    return flag

def check_discourse_markers(str):
    flag = False
    if str[1] == 'C' or str[1] == 'Cc':
        flag = True
    return flag

def get_entity(doc):
    l = []
    for item in doc:
        if item[1] == 'Np' or item[1] == 'Y':
            l.append(item[0].lower())
    return l

def compare_name_entity(str1, str2):
    flag = False
    doc1 = get_entity(str1)
    doc2 = get_entity(str2)
    if len(doc1)>0 and len(doc2)>0:
        for text in doc1:
            if text in doc2:
                flag=True
                break
    return flag

def segment(str_text):
    doc = vn_pos_tag.annotate_text(str_text)
    textlist = []
    for item in doc[0]:
        source_token = item.get('wordForm')
        source_pos = item.get('posTag')
        if source_pos == 'CH':
            source_pos = source_token
        textlist.append((source_token, source_pos))
    return textlist