from kiwipiepy import Kiwi
kiwi = Kiwi()

class Tokenizer():
        
    def __call__(self, data):
        list = []
        
        p = kiwi.tokenize(data)
        
        for i in range(len(p)):
            if p[i].tag == 'NNP':
                if p[i].form.endswith('특별시') == True or p[i].form.endswith('광역시') == True:
                    list.append(p[i].form[:-3])
                
                elif p[i].form.endswith('시') == True or p[i].form.endswith('군') == True or p[i].form.endswith('구') == True:
                    list.append(p[i].form[:-1])
                    
                elif p[i].form.endswith('동') == True or p[i].form.endswith('면') == True or p[i].form.endswith('읍') == True or p[i].form.endswith('리') == True:
                    list.append(p[i].form[:-1])
                
                elif p[i].form.endswith('역') == True:
                    list.append(p[i].form[:-1])
                
                elif p[i].form.endswith('학교') == True:
                    list.append(p[i].form[:-2])
                    
                elif p[i].form.endswith('교') == True:
                    list.append(p[i].form[:-1])
               
                else:
                    list.append(p[i].form)
                
        return ( " ".join( list ))