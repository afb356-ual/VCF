'''Codificación de Entropía de imágenes usando Arithmetic Coding'''
import io
import os
import sys
import gzip
import pickle
import logging
import numpy as np
from bitarray import bitarray

# Preservar documentación
with open("/tmp/description.txt", 'w') as f:
    f.write(__doc__)

import main
import parser
import entropy_image_coding as EIC

class CoDec(EIC.CoDec):
    def __init__(self, args):
        super().__init__(args)
        self.file_extension = ".arith"
        self.BITS = 32
        self.MAX = 1 << self.BITS
        self.HALF = self.MAX >> 1
        self.QUARTER = self.HALF >> 1
        self.THREE_QUARTER = self.QUARTER * 3
        
        # --- SOLUCIÓN: Usar potencia de 2 (2^16 = 65536) ---
        # Esto permite sustituir la división lenta (/) por bitshift rápido (>>)
        # y evita los errores de redondeo que rompen la imagen.
        self.PRECISION_BITS = 16 
        self.TOTAL_FREQ = 1 << self.PRECISION_BITS

    def _normalize_counts(self, raw_counts):
        """Escala las frecuencias para que sumen exactamente 65536"""
        total_raw = np.sum(raw_counts)
        if total_raw == 0: return raw_counts
        
        scale = self.TOTAL_FREQ / total_raw
        new_counts = np.floor(raw_counts * scale).astype(np.int32)
        new_counts[new_counts == 0] = 1 # Evitar frecuencias 0
        
        # Ajuste fino para que la suma sea exacta
        diff = self.TOTAL_FREQ - np.sum(new_counts)
        new_counts[np.argmax(new_counts)] += diff
        
        return new_counts

    def compress_fn(self, img, fn):
        stats_fn = f"{fn}_arith_stats.pkl.gz"
        compressed_img = io.BytesIO()

        flat_img = img.flatten().astype(np.int32)
        symbols, raw_counts = np.unique(flat_img, return_counts=True)
        
        # 1. NORMALIZAR (Clave para velocidad y precisión)
        counts = self._normalize_counts(raw_counts)
        
        highs = np.cumsum(counts)
        lows = np.roll(highs, 1)
        lows[0] = 0
        
        freq_map = {int(s): (int(l), int(h)) for s, l, h in zip(symbols, lows, highs)}

        # 2. Codificar
        bit_stream = self._encode(flat_img, freq_map)

        compressed_img.write(bit_stream.tobytes())

        with gzip.open(stats_fn, 'wb') as f:
            np.save(f, img.shape)
            pickle.dump({'symbols': symbols, 'counts': counts}, f)

        # TRADUCIDO AL CASTELLANO
        logging.info(f"Tamaño comprimido: {len(bit_stream)} bits")
        return compressed_img

    def decompress_fn(self, compressed_bytes, fn):
        stats_fn = f"{fn}_arith_stats.pkl.gz"
        
        with gzip.open(stats_fn, 'rb') as f:
            shape = np.load(f)
            stats = pickle.load(f)
        
        symbols = stats['symbols']
        counts = stats['counts']

        # --- TABLAS RÁPIDAS ---
        # Tabla inversa (Lookup Table) para decodificación O(1)
        lookup_table = np.repeat(symbols, counts).astype(np.uint8)
        lookup_bytes = lookup_table.tobytes()

        list_lows = [0] * 256
        list_highs = [0] * 256
        
        current_low = 0
        for sym, count in zip(symbols, counts):
            s = int(sym)
            list_lows[s] = current_low
            current_low += int(count)
            list_highs[s] = current_low

        bit_stream = bitarray(endian='big')
        bit_stream.frombytes(io.BytesIO(compressed_bytes).read())

        decoded_data = self._decode(
            bit_stream, 
            list_lows, 
            list_highs, 
            lookup_bytes, 
            int(np.prod(shape))
        )

        return np.array(decoded_data).reshape(shape).astype(np.uint8)

    def _encode(self, data, freq_map):
        low = 0
        high = self.MAX
        pending_bits = 0
        output = bitarray(endian='big')

        HALF = self.HALF
        QUARTER = self.QUARTER
        THREE_QUARTER = self.THREE_QUARTER
        SHIFT = self.PRECISION_BITS 

        for symbol in data:
            s_low, s_high = freq_map[symbol]
            range_ = high - low
            
            high = low + ((range_ * s_high) >> SHIFT)
            low  = low + ((range_ * s_low)  >> SHIFT)

            while True:
                if high <= HALF:
                    self._emit_bit(output, 0, pending_bits)
                    pending_bits = 0
                elif low >= HALF:
                    self._emit_bit(output, 1, pending_bits)
                    pending_bits = 0
                    low -= HALF
                    high -= HALF
                elif low >= QUARTER and high <= THREE_QUARTER:
                    pending_bits += 1
                    low -= QUARTER
                    high -= QUARTER
                else:
                    break
                low *= 2
                high *= 2
        
        pending_bits += 1
        if low < QUARTER:
            self._emit_bit(output, 0, pending_bits)
        else:
            self._emit_bit(output, 1, pending_bits)
            
        return output

    def _decode(self, bits, list_lows, list_highs, lookup_bytes, num_symbols):
        low = 0
        high = self.MAX
        value = 0
        
        bit_list = bits.tolist()
        bit_iter = iter(bit_list)
        
        for _ in range(self.BITS):
            value <<= 1
            if next(bit_iter, 0): value |= 1

        decoded = bytearray(num_symbols)
        
        HALF = self.HALF
        QUARTER = self.QUARTER
        THREE_QUARTER = self.THREE_QUARTER
        SHIFT = self.PRECISION_BITS 
        TOTAL_FREQ = self.TOTAL_FREQ

        log_step = num_symbols // 5

        for i in range(num_symbols):
            if i % log_step == 0:
                logging.info(f"Decodificando... {i}/{num_symbols}")
            range_ = high - low
            
            # Calculamos el offset escalado a 16 bits
            offset = value - low
            # ((value - low) * TOTAL) // range
            scaled_value = ((offset + 1) << SHIFT) - 1
            scaled_value = scaled_value // range_
            
            if scaled_value >= TOTAL_FREQ:
                scaled_value = TOTAL_FREQ - 1

            # Búsqueda O(1)
            symbol = lookup_bytes[scaled_value]
            decoded[i] = symbol

            s_low = list_lows[symbol]
            s_high = list_highs[symbol]
        
            high = low + ((range_ * s_high) >> SHIFT)
            low  = low + ((range_ * s_low)  >> SHIFT)

            while True:
                if high <= HALF:
                    pass
                elif low >= HALF:
                    low -= HALF
                    high -= HALF
                    value -= HALF
                elif low >= QUARTER and high <= THREE_QUARTER:
                    low -= QUARTER
                    high -= QUARTER
                    value -= QUARTER
                else:
                    break
                
                low += low
                high += high
                value += value
                
                try:
                    if next(bit_iter): value |= 1
                except StopIteration:
                    pass

        return decoded

    def _emit_bit(self, output, bit, pending):
        output.append(bit)
        if pending:
            output.extend([not bit] * pending)

    def compress(self, img, fn="/tmp/encoded"): return self.compress_fn(img, fn)
    def decompress(self, compressed_img, fn="/tmp/encoded"): return self.decompress_fn(compressed_img, fn)

if __name__ == "__main__":
    main.main(parser.parser, logging, CoDec)