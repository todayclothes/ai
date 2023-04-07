from kiwipiepy import Kiwi
kiwi = Kiwi()

class Tokenizer():
    def __call__(self, data):
        list = []
        
        p = kiwi.tokenize(data)
        
        for i in range(len(p)):
            if p[i].tag.startswith('J') == False:
                list.append(p[i].form)
                
        return ( " ".join( list ))