
# coding: utf-8



import pandas as pd
import re




def filter(s, match):
    begin = match.start(0)
    length = len(match.group())
    str_1 = ""
    for word in s[:begin]:
        str_1 += word
    for word in s[begin+length:]:
        str_1 += word   
    return str_1




df1 = pd.read_csv('5.csv',names = ['app_id','app_name','ratingValue','ratingCount','permission','description'])
data = df1.description
pattern = re.compile(r'(((https?|ftp|file)://)|www)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')




g = open('filter.csv', 'w+')

for i in data:

    try:
        if len(i) >10:
            match = pattern.search(i)
            if not match:
                str_1 = ""
                for word in i:
                    str_1 += word
            else:
                
                while match:
                    i = filter(i, match)
                    match = pattern.search(i)
                    
                str_1 = i
        g.write(str_1)
        g.write('\n' )
    except:
        pass
g.close()

