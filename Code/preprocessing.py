# Converting from inkml to png
import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image

def parse_inkml(file_path):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        traces = {}
        for trace in root.findall('{http://www.w3.org/2003/InkML}trace'):
            # trace_id = trace.attrib['id']
            trace_id = str(trace.attrib['id'])

            coords = []
            for point in trace.text.strip().split(','):
                if point.strip() == '':
                    continue
                parts = point.strip().split()
                if len(parts) >= 2:
                    x = float(parts[0])
                    y = float(parts[1])
                    coords.append((x, y))
            traces[trace_id] = coords

        trace_groups = []
        for traceView in root.findall('.//{http://www.w3.org/2003/InkML}traceView'):
            trace_groups.append(traceView.attrib['traceDataRef'])

        annotation = root.find('{http://www.w3.org/2003/InkML}annotation')
        label = annotation.text if annotation is not None else ""
        return traces, trace_groups, label
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None, None, None

def normalize_and_render(traces, trace_groups=None, img_size=256, padding=10):
    all_points = []
    keys = trace_groups if trace_groups else traces.keys()

    for t_id in keys:
        if t_id in traces:  # <-- fix here
            all_points.extend(traces[t_id])
    
    if not all_points:
        raise ValueError("No valid stroke points found.")

    all_points = np.array(all_points)
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    scale = (img_size - 2 * padding) / max(max_x - min_x, max_y - min_y + 1e-6)
    canvas = np.ones((img_size, img_size), dtype=np.uint8) * 255

    for t_id in keys:
        if t_id in traces:
            trace = np.array(traces[t_id])
            trace -= [min_x, min_y]
            trace *= scale
            trace += padding
            trace = trace.astype(np.int32)

            for i in range(1, len(trace)):
                pt1 = tuple(trace[i - 1])
                pt2 = tuple(trace[i])
                cv2.line(canvas, pt1, pt2, color=0, thickness=2)

    return canvas


def process_inkml_folder(inkml_root_dir, user_output_dir):
    output_img_dir = os.path.join(user_output_dir, "processed_images")
    output_label_path = os.path.join(user_output_dir, "labels.txt")

    os.makedirs(output_img_dir, exist_ok=True)

    with open(output_label_path, 'w', encoding='utf-8') as label_file:
        for root, _, files in os.walk(inkml_root_dir):
            for file in files:
                if file.endswith('.inkml'):
                    file_path = os.path.join(root, file)
                    traces, trace_groups, label = parse_inkml(file_path)
                    if traces:
                        img = normalize_and_render(traces, trace_groups)
                        base_name = os.path.splitext(file)[0]
                        img_path = os.path.join(output_img_dir, f"{base_name}.png")
                        Image.fromarray(img).save(img_path)
                        label_file.write(f"{base_name}.png\t{label}\n")
                        print(f"Processed: {file}")


inkml_dir = '/kaggle/input/crohme2019/crohme2019/crohme2019/valid'
user_output_dir = 'results_valid'


process_inkml_folder(inkml_dir, user_output_dir)
print(f"\nDone! Processed images and labels saved in: {user_output_dir}")

# Tokenizing (inspired from https://www.kaggle.com/code/hayamonawwar/crohme-ctc-pytorch-cur/edit)
input_files = ['/kaggle/input/crohme2019/crohme2019_train.txt',
'/kaggle/input/crohme2019/crohme2019_valid.txt',
'/kaggle/input/crohme2019/crohme2019_test.txt']

vocab = set()

for input_file in input_files:
    for line in open(input_file).readlines():
        if len(line.strip().split('\t')) == 2:
            vocab.update(line.strip().split('\t')[1].split())
vocab_syms = [v for v in vocab if v not in ['Above', 'Below', 'Inside', 'NoRel', 'Right', 'Sub', 'Sup']]

with open('crohme_vocab.txt', 'w') as f:
    f.writelines([c + '\n' for c in sorted(vocab_syms)])
    f.writelines([c + '\n' for c in ['Above', 'Below', 'Inside', 'NoRel', 'Right', 'Sub', 'Sup']])

class Vocab(object):
    def __init__(self, vocab_file=None):
        self.word2index = {}
        self.index2word = {}

        if vocab_file:
            self.load_vocab(vocab_file)
    
    def load_vocab(self, vocab_file):
        # load vocab from file
        with open(vocab_file, 'r') as f:
            for i, line in enumerate(f):
                word = line.strip()
                self.word2index[word] = i
                self.index2word[i] = word
        # add blank word
        self.word2index['<blank>'] = len(self.word2index)
        self.index2word[self.word2index['<blank>']] = '<blank>'

vocab = Vocab(vocab_file = '/kaggle/working/crohme_vocab.txt')
vocab.index2word

# Testing
vocab = Vocab('crohme_vocab.txt')
input = '- Right \\sqrt Inside 2'.split()
output = [vocab.word2index[word] for word in input]
output # Shpuld be [4, 105, 66, 103, 9]
