

peta_insertname=['<ImageHere_Head>', 
                 '<ImageHere_Top>', 
                 '<ImageHere_Under>', 
                 '<ImageHere_Shoes>', 
                 '<ImageHere_Bag>', 
                 '<ImageHere_Face>']
peta_instuction= '''###Human: Analyse a person's photo and use the options provided to categorise their appearance into attributes, 
<Img><ImageHere_Head></Img> What wearing on head? 
<Img><ImageHere_Top></Img> What wearing on top? 
<Img><ImageHere_Under></Img> What wearing on under? 
<Img><ImageHere_Shoes></Img> What type about shoes? 
<Img><ImageHere_Bag></Img> What type about bag? 
<Img><ImageHere_Face></Img> How old? Is man or woman?. ###Assistant:'''
peta_group_details = [5,10,6,4,5,5]
# print(peta_instuction.replace('\n', ''))
pa100k_insertname=['<ImageHere_Face>', 
                   '<ImageHere_Occ>', 
                   '<ImageHere_Head>', 
                   '<ImageHere_Bag>', 
                   '<ImageHere_Top>', 
                   '<ImageHere_Under>']
pa100k_instuction= '''
###Human: Analyse a person's photo and use the options provided to categorise their appearance into attributes, 
<Img><ImageHere_Face></Img> Is man or woman? How old? 
<Img><ImageHere_Occ></Img> Where is the person facing to? 
<Img><ImageHere_Head></Img> What wearing on head? 
<Img><ImageHere_Bag></Img> What is the topwear? 
<Img><ImageHere_Top></Img> What type about bag? 
<Img><ImageHere_Under></Img> What is the bottomwear? Have boots? ###Assistant:'''
pa100k_group_details = [4,3,2,4,6,7]

rap_insertname=['<ImageHere_Head>', 
                '<ImageHere_Top>', 
                '<ImageHere_Under>', 
                '<ImageHere_Shoes>', 
                '<ImageHere_Bag>', 
                '<ImageHere_Face>', 
                '<ImageHere_Full>']
rap1_instuction='''
###Human: Analyze the photo,<Img><ImageHere_Head></Img> What is on the head? 
<Img><ImageHere_Top></Img> What is the topwear? 
<Img><ImageHere_Under></Img> What is the bottomwear? 
<Img><ImageHere_Shoes></Img> What type of shoes? 
<Img><ImageHere_Bag></Img> What type of bag? 
<Img><ImageHere_Face></Img> Is the person male or female? Age? 
<Img><ImageHere_Full></Img> Body shape? Identity? Activity? ###Assistant:'''
rap1_group_details=[6,9,6,5,8,4,13]

rap2_instuction='''
###Human: Analyze the photo,
<Img><ImageHere_Head></Img> What is on the head? 
<Img><ImageHere_Top></Img> What is the topwear? 
<Img><ImageHere_Under></Img> What is the bottomwear? 
<Img><ImageHere_Shoes></Img> What type of shoes? 
<Img><ImageHere_Bag></Img> What type of bag? 
<Img><ImageHere_Face></Img> Age? Is the person male or female? 
<Img><ImageHere_Full></Img> Body shape? Identity? Activity? ###Assistant:'''

rap2_group_details=[5,10,6,6,8,5,14]

msp_insertname=['<ImageHere_Head>', 
                '<ImageHere_Top>', 
                '<ImageHere_Under>', 
                '<ImageHere_Shoes>', 
                '<ImageHere_Bag>', 
                '<ImageHere_Face>', 
                '<ImageHere_Full>']
msp_instuction='''
###Human: Analyze the photo,
<Img><ImageHere_Head></Img> What is on the head? 
<Img><ImageHere_Top></Img> What is the topwear? 
<Img><ImageHere_Under></Img> What is the bottomwear? 
<Img><ImageHere_Shoes></Img> What type of shoes? 
<Img><ImageHere_Bag></Img> What type of bag? 
<Img><ImageHere_Face></Img> Age? Is the person male or female? 
<Img><ImageHere_Full></Img> Body shape? Facing to where? Activity? ###Assistant:'''
msp_group_details=[9,10,6,5,7,4,16]
instruction={
    "PETA":{
        "instruction":peta_instuction,
        "insert_name":peta_insertname,
        "group_details":peta_group_details,
            },
    "PA100k":{
        "instruction":pa100k_instuction,
        "insert_name":pa100k_insertname,
        "group_details":pa100k_group_details,
            },
    "RAPv1":{
        "instruction":rap1_instuction,
        "insert_name":rap_insertname,
        "group_details":rap1_group_details,
            },
    "RAPv2":{
        "instruction":rap2_instuction,
        "insert_name":rap_insertname,
        "group_details":rap2_group_details,
            },
    "MSP":{
        "instruction":msp_instuction,
        "insert_name":msp_insertname,
        "group_details":msp_group_details,
            },
    "MSPCD":{
        "instruction":msp_instuction,
        "insert_name":msp_insertname,
        "group_details":msp_group_details,
            }
}