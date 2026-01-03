'''Image blurring using low-pass filtering. *** Effective only when decoding! ***'''

import numpy as np
import logging
import parser
import main
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)
#import entropy_image_coding as EIC
import importlib
import cv2

#import entropy_image_coding as EIC
#import importlib


default_filter_size = 5
#default_blur_filter = "gaussian"
#default_EIC = "TIFF"

#_parser, parser_encode, parser_decode = parser.create_parser(description=__doc__)

# Encoder parser
#parser.parser_encode.add_argument("-c", "--entropy_image_codec", help=f"Entropy Image Codec (default: {default_EIC})", default=default_EIC)

# Decoder parser
#parser.parser_decode.add_argument("-f", "--blur_filter", help=f"Blurring filter name (gaussian, median or blur) (default: {default_blur_filter})", default=default_blur_filter)
parser.parser_decode.add_argument("-s", "--filter_size", type=parser.int_or_str, help=f"Filter size (default: {default_filter_size})", default=default_filter_size)
import no_filter

args = parser.parser.parse_known_args()[0]
#EC = importlib.import_module(args.entropy_image_codec)

class CoDec(no_filter.CoDec):

    def __init__(self, args):
        logging.debug(f"trace args={args}")
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        self.args = args
        #if self.encoding:
        #    self.filter = "gaussian"
        #    self.filter_size = 0

    def decode(self):
        compressed_k = self.decode_read()
        k = self.decompress(compressed_k)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")        
        y = self.filter(k)
        output_size = self.decode_write(y)
        return output_size
            
    def filter(self, img):
        logging.debug(f"trace y={img}")
        logging.info(f"filter size={self.args.filter_size}")
        return cv2.GaussianBlur(img, (self.args.filter_size, self.args.filter_size), 0)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)
