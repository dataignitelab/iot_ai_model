from dataset import load_image
from tqdm import tqdm

if __name__ == '__main__':
    data_path = '/home/workspace/iot_ai_model/dataset/supervisely_person/test_data_list.txt'
    with open(data_path, 'r') as f:
        line = f.readlines()

    base = '/home/workspace/iot_ai_model/dataset/supervisely_person/'

    for p in tqdm(line):
        filename = p.split(',')
        load_image(base + filename[0])