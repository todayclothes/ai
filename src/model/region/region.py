from mecab import MeCab
mecab = MeCab()

class region_Tokenizer():
        
    def __call__(self, data):
        list = []
        
        p = mecab.nouns(data)
        
        for i in range(len(p)):
            if p[i].endswith('특별시') == True or p[i].endswith('광역시') == True:
                list.append(p[i][:-3])
                
            elif p[i].endswith('시') == True or p[i].endswith('군') == True or p[i].endswith('구') == True:
                list.append(p[i][:-1])
                    
            elif p[i].endswith('동') == True or p[i].endswith('면') == True or p[i].endswith('읍') == True or p[i].endswith('리') == True:
                list.append(p[i][:-1])
                
            elif p[i].endswith('역') == True:
                list.append(p[i][:-1])
                
            elif p[i].endswith('학교') == True:
                list.append(p[i][:-2])
               
            else:
                list.append(p[i])
                
        return ( " ".join( list ))