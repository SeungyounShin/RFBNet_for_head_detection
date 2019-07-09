# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 08:28:53 2018

@author: Peng Dezhi
"""

import xml.etree.ElementTree as ET
import os

def valid_annotation_label(filename, labelname):
    tree = ET.parse(filename)
    objs = tree.findall('object')
    
    for obj in objs:
        if not obj.find('name').text == labelname:
            return False
        
    return True

if __name__ == '__main__':
    ano_dir = './Annotations'
    filename_list = os.listdir(ano_dir)
    for filename in filename_list:
        filepath = os.path.join(ano_dir, filename)
        if not valid_annotation_label(filepath, 'person'):
            print(filename)
            
            