import argparse
import time
import requests
from PIL import Image
from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                         help='File path of .tflite file.')
    parser.add_argument('-i', '--input', required=True,
                          help='Image to be classified.')
    parser.add_argument('-l', '--labels',
                      help='File path of labels file.')
    parser.add_argument('-k', '--top_k', type=int, default=1,
                      help='Max number of classification results')
    parser.add_argument('-t', '--threshold', type=float, default=0.0,
                      help='Classification score threshold')
    parser.add_argument('-c', '--count', type=int, default=5,
                      help='Number of times to run inference')
    args = parser.parse_args()

    labels = read_label_file(args.labels) if args.labels else {}

    interpreter = make_interpreter(*args.model.split('@'))
    interpreter.allocate_tensors()

    size = common.input_size(interpreter)
    image = Image.open(args.input).convert('RGB').resize(size, Image.ANTIALIAS)
    common.set_input(interpreter, image)

    for _ in range(args.count):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        classes = classify.get_classes(interpreter, args.top_k, args.threshold)

    for c in classes:
        print('%s' % (labels.get(c.id, c.id)))
        
    params={'location':'python requeststest','time_of_day':'dinner'}
    params['rate'] =  int(labels.get(c.id, c.id))
    requests.get(url="https://j85andhx7b.execute-api.us-east-1.amazonaws.com/dev/createTest",params=params)

if __name__ == '__main__':
    main()
