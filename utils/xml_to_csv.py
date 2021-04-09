import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from config import config

def xml_to_csv(path):
    count = 0
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        if len(root.findall('part'))>0:
            continue
        for member in root.findall('object'):
            if member[0].text != 'ring':
                continue
            try:
                test = int(float(member[4][0].text))
                test = int(float(member[4][1].text))
                test = int(float(member[4][2].text))
                test = int(float(member[4][3].text))
            except:
                continue
            count+=1
            print("Example number: {}".format(count))
            filename = root.find('path').text
            bndbox = member.find('bndbox')
            value = (filename,
                     member[0].text,
                     int(float(bndbox[0].text)),
                     int(float(bndbox[1].text)),
                     int(float(bndbox[2].text)),
                     int(float(bndbox[3].text))
                     )
            xml_list.append(value)
    column_name = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


if __name__ == "__main__":
    xml_path = os.path.sep.join([r"D:\\cocosynth\\datasets\\ring_dataset\\output\\xml"])
    xml_df = xml_to_csv(xml_path)
    xml_df.to_csv('D:\\cocosynth\\datasets\\ring_dataset\\output\\csv\\ring.csv', index=None)
    print('Successfully converted xml to csv.')
