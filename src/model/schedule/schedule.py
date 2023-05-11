from mecab import MeCab
mecab = MeCab()

class schedule_Tokenizer():
        
    def __call__(self, data):
        list = []
        
        p = mecab.pos(data)
        
        for i in range(len(p)):
            if p[i][1].startswith('J') == False:
                list.append(p[i][0])
                
        return ( " ".join( list ))