# -*- coding: UTF-8 -*-
import os 
import os.path 
import json 
import string

rootdir = "googleApps"

fname = [] 

for parent,dirnames,filenames in os.walk(rootdir):
    for filename in filenames: 
        fname.append(filename)

g = open('5.csv', 'w+')

for name in fname: 
    with open('googleApps/' + str(name), 'r') as f: 
        json1 = json.load(f) 
        n = json1['name']#.encode('utf-8')
        n0 = ""
        for ch in n:
            try:
                n0 = n0 + str(ch)
            except:
                pass
        
        d = json1['description']#.encode('utf-8')
        d0 = ""
        for ch in d:
            try:
                d0 = d0 + str(ch)
            except:
                pass
        
        g.write(name) 
        g.write(',') 
        n1 = n0.replace(',',' ')         
        g.write(n1) 
        g.write(',') 
        g.write(str(json1['ratingValue'])) 
        g.write(',') 
        g.write(str(json1['ratingCount'])) 
        g.write(',') 
        p = " &".join(json1['permission'])
        g.write(str(p)) 
        g.write(',')
        d1 = d0.replace(',','#')
        g.write(d1)
        g.write('\n' )        
g.close()