import os
import marshal
import pickle
import array
import struct
from collections import defaultdict

# based on: http://code.activestate.com/recipes/576603-huffman-coding-encoderdeconder/

class HuffmanNode(object):
    recurPrint = False
    def __init__(self, symbol=None, freq=0, left=None, right=None, parent=None):
        self.left = left
        self.right = right
        self.parent = parent
        self.symbol = symbol
        self.freq = freq


    def __repr__(self):
        if HuffmanNode.recurPrint:
            left = self.left if self.left else '#'
            right = self.right if self.right else '#'

            if self.is_leaf():
                return ''.join( ('(%s:%d)' % (self.symbol, self.freq), str(left), str(right) ) )
            else:
                return ''.join((str(left), str(right)))
        else:
            return '(%s:%d)'%(self.symbol, self.freq)


    def __lt__(self, other):
        if not isinstance(other, HuffmanNode):
            return super(HuffmanNode, self).__lt__(other)
        return self.freq < other.freq


    def __eq__(self, other):
        if not isinstance(other, HuffmanNode):
            return super(HuffmanNode, self).__eq__(other)
        return self.freq == other.freq

    def is_leaf(self):
        return (not self.left and not self.right)


def huffman_to_bytes(huffman):
    # return pickle.dumps(huffman)
    return pickle.dumps(_huffman_to_bytes(huffman))

def _huffman_to_bytes(huffman):
    if huffman.is_leaf():
        return huffman.symbol
    else:
        return [_huffman_to_bytes(huffman.left), _huffman_to_bytes(huffman.right)]


def huffman_from_bytes(bytes):
    huffman_array = pickle.loads(bytes)

    if isinstance(huffman_array, list):
        return _huffman_from_bytes(huffman_array)
    else:
        return HuffmanNode(symbol=huffman_array)

def _huffman_from_bytes(huffman_array):
    if isinstance(huffman_array[0], list):
        left = _huffman_from_bytes(huffman_array[0])
    else:
        left = HuffmanNode(symbol=huffman_array[0])

    if isinstance(huffman_array[1], list):
        right = _huffman_from_bytes(huffman_array[1])
    else:
        right = HuffmanNode(symbol=huffman_array[1])

    return HuffmanNode(left=left, right=right)


def _build_tree(nodes):
    nodes.sort()
    while(True):
        if len(nodes) == 1:
            return nodes[0]

        first = nodes.pop(0)
        second = nodes.pop(0)

        parent = HuffmanNode(left=first, right=second, freq=first.freq+second.freq)
        first.parent = parent
        second.parent = parent

        nodes.insert(0, parent)
        nodes.sort()


def _gen_huffman_code(node, dict_codes, buffer_stack=[]):
    if node.is_leaf():
        dict_codes[node.symbol] = ''.join(buffer_stack)
        return

    buffer_stack.append('0')
    _gen_huffman_code(node.left, dict_codes, buffer_stack)
    buffer_stack.pop()

    buffer_stack.append('1')
    _gen_huffman_code(node.right, dict_codes, buffer_stack)
    buffer_stack.pop()


def _cal_freq(data):
    d = defaultdict(int)
    for c in data:
        d[c] += 1
    return d


MAX_BITS = 8

class Encoder(object):
    def __init__(self, long_str):
        self.long_str = long_str
        # if filename_or_long_str:
        #     if os.path.exists(filename_or_long_str):
        #         self.encode(filename_or_long_str)
        #     else:
        #         print('[Encoder] take \'%s\' as a string to be encoded.' % filename_or_long_str)
        #         self.long_str = filename_or_long_str


    def __get_long_str(self):
        return self._long_str


    def __set_long_str(self, s):
        self._long_str = s
        if s:
            self.root = self._get_tree_root()
            self.code_map = self._get_code_map()
            self.array_codes, self.code_length = self._encode()
    long_str = property(__get_long_str, __set_long_str)


    def _get_tree_root(self):
        d = _cal_freq(self.long_str)
        return _build_tree([HuffmanNode(ch=ch, fq=int(fq)) for ch, fq in d.items()])


    def _get_code_map(self):
        a_dict={}
        _gen_huffman_code(self.root, a_dict)
        return a_dict


    def _encode(self):
        array_codes = array.array('B')
        code_length = 0
        buff, length = 0, 0
        for ch in self.long_str:
            code = self.code_map[ch]
            for bit in list(code):
                if bit=='1':
                    buff = (buff << 1) | 0x01
                else: # bit == '0'
                    buff = (buff << 1)
                length += 1
                if length == MAX_BITS:
                    array_codes.extend([buff])
                    buff, length = 0, 0

            code_length += len(code)

        if length != 0:
            array_codes.extend([buff << (MAX_BITS-length)])

        return array_codes, code_length


    # def encode(self, filename):
    #     fp = open(filename, 'rb')
    #     self.long_str = fp.read()
    #     fp.close()


    def write(self, filename):
        if self._long_str:
            fcompressed = open(filename, 'wb')
            marshal.dump((pickle.dumps(self.root), self.code_length, self.array_codes), fcompressed)
            fcompressed.close()
        else:
            print("You haven't set 'long_str' attribute.")


class Decoder(object):
    def __init__(self, raw_str):
        # if filename_or_raw_str:
        #     if os.path.exists(filename_or_raw_str):
        #         filename = filename_or_raw_str
        #         self.read(filename)
        #     else:
        #         print('[Decoder] take \'%s\' as raw string' % filename_or_raw_str)
        #         raw_string = filename_or_raw_str
        #         unpickled_root, length, array_codes = marshal.loads(raw_string)
        #         self.root = pickle.loads(unpickled_root)
        #         self.code_length = length
        #         self.array_codes = array.array('B', array_codes)
        unpickled_root, length, array_codes = marshal.loads(raw_str)
        self.root = pickle.loads(unpickled_root)
        self.code_length = length
        self.array_codes = array.array('B', array_codes)


    def decode(self):
        string_buf = []
        total_length = 0
        node = self.root
        for code in self.array_codes:
            buf_length = 0
            while (buf_length < MAX_BITS and total_length != self.code_length):
                buf_length += 1
                total_length += 1
                if code >> (MAX_BITS - buf_length) & 1:
                    node = node.R
                    if node.c:
                        string_buf.append(node.c)
                        node = self.root
                else:
                    node = node.L
                    if node.c:
                        string_buf.append(node.c)
                        node = self.root

        # for v in string_buf:
        #     print(chr(v))
        return bytes(string_buf)


    def read(self, filename):
        fp = open(filename, 'rb')
        unpickled_root, length, array_codes = marshal.load(fp)
        self.root = pickle.loads(unpickled_root)
        self.code_length = length
        self.array_codes = array.array('B', array_codes)
        fp.close()


    def decode_as(self, filename):
        decoded = self.decode()
        fout = open(filename, 'wb')
        fout.write(decoded)
        fout.close()


def encode(data):
    freq_dict = _cal_freq(data)
    huffman_tree = _build_tree([HuffmanNode(symbol=symbol, freq=int(freq)) for symbol, freq in freq_dict.items()])
    huffman_code = {}
    _gen_huffman_code(huffman_tree, huffman_code)

    encoded_data = array.array('B')
    code_length = 0

    buff = 0
    buff_length = 0
    for symbol in data:
        code = huffman_code[symbol]
        for bit in code:
            buff <<= 1
            if bit == '1':
                buff |= 0x01

            buff_length += 1
            if buff_length == MAX_BITS:
                encoded_data.append(buff)
                buff = 0
                buff_length = 0

        code_length += len(code)

    if buff_length != 0:
        encoded_data.append(buff << (MAX_BITS - buff_length))

    bytes_tree = huffman_to_bytes(huffman_tree)
    length_tree = len(bytes_tree)
    length_data = len(encoded_data)

    # print('Huff: %d | Data: %d' % (length_tree, length_data))

    fmt = 'iii%us%us' % (length_tree, length_data)

    return struct.pack(fmt, length_tree, length_data, code_length, bytes_tree, bytes(encoded_data))


def decode(data):
    (length_tree, length_data, code_length), data = struct.unpack('iii', data[:12]), data[12:]

    fmt = '%us%us' % (length_tree, length_data)
    bytes_tree, encoded_data = struct.unpack(fmt, data)

    huffman_tree = huffman_from_bytes(bytes_tree)

    decoded_data = []
    total_length = 0

    node = huffman_tree
    for code in encoded_data:
        buff_length = 0
        while (buff_length < MAX_BITS and total_length != code_length):
            buff_length += 1
            total_length += 1

            if code & (1 << (MAX_BITS - buff_length)):
                node = node.right
            else:
                node = node.left

            if node.is_leaf():
                decoded_data.append(node.symbol)
                node = huffman_tree

    return bytes(decoded_data)


if __name__=='__main__':
    original_file = 'huff1.txt'
    compressed_file = 'huff2.scw'
    decompressed_file = 'huff3.txt'

    # first way to use Encoder/Decoder
    enc = Encoder(original_file)
    enc.write(compressed_file)
    dec = Decoder(compressed_file)
    dec.decode_as(decompressed_file)

    # second way
    #enc = Encoder()
    #enc.encode(original_file)
    #enc.write(compressed_file)
    #dec = Decoder()
    #dec.read(compressed_file)
    #dec.decode_as(decompressed_file)
