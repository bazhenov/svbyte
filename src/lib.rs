/*!
This library provides encoding/decoding primitives for Stream VByte encoding.

Stream VByte encoding is a SIMD accelerated algorithm of VarInt decompression. It is used in a search and database
systems as a way of efficiently store and stream large number of VarInts from a disk or main memory.

The idea behind VarInt is not to store leading bytes of a number, so large amount of relatively small numbers can be
stored in a much more compact way. VarInt encoding is frequently used with delta-encoding if numbers are stored in the
ascending order. This way all the numbers are smaller by magnitude, hence better compression.

Stream VByte working using two data streams: control stream and data stream. Control stream contains control words (1
byte each). Each control word describe length of 4 numbers in the data stream (2 bits per number, 00 - length 1, 01 -
length 2 and so on).

- [Decoding billions of integers per second through vectorization][pub] by Daniel Lemire and Leonid Boytsov.
- [Stream VByte: breaking new speed records for integer compression][blog-post] by Daniel Lemire

[pub]: https://arxiv.org/abs/1209.2137
[blog-post]: https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/
*/
use std::io::{self, Write};

pub const MASKS: [u128; 256] = shuffle_masks();

/// Stream VByte Encoder
pub struct StreamVByteEncoder<W> {
    data_stream: Box<W>,
    control_stream: Box<W>,
    control_word: u8,
    written: u8,
}

impl<W: Write> StreamVByteEncoder<W> {
    pub fn new(data: W, control: W) -> Self {
        let data = Box::new(data);
        let control = Box::new(control);
        Self {
            data_stream: data,
            control_stream: control,
            control_word: 0,
            written: 0,
        }
    }
    /// Compresses input data using stream algorithm
    pub fn encode(&mut self, input: &[u32]) -> io::Result<()> {
        for n in input {
            let bytes: [u8; 4] = n.to_be_bytes();
            let length = 4 - n.leading_zeros() as u8 / 8;
            assert!(length <= 4);

            self.control_word <<= 2;
            self.control_word |= length - 1;

            self.data_stream.write_all(&bytes[4 - length as usize..])?;
            self.written += 1;
            self.ensure_control_word_written()?;
        }
        Ok(())
    }

    fn ensure_control_word_written(&mut self) -> io::Result<()> {
        if self.written == 4 {
            self.control_stream.write_all(&[self.control_word])?;
            self.control_word = 0;
            self.written = 0;
        }
        Ok(())
    }

    /// Returns control and data stream back to the client
    ///
    /// Flushes all pending writes and returns tuple of two streams `(control, data)`.
    pub fn finish(mut self) -> io::Result<(W, W)> {
        // We need to pad last control word with zero bits if number of elements
        // not multiple of 4, otherwise last control word will not be written
        if self.written > 0 {
            self.control_word <<= 2 * (4 - self.written);
            self.written = 4;
        }

        self.ensure_control_word_written()?;
        self.data_stream.flush()?;
        self.control_stream.flush()?;

        Ok((*self.control_stream, *self.data_stream))
    }
}

/// Represents an object that can decode a stream of data into a buffer of fixed size. A type parameter `T` specifies /// the type of the elements in the buffer, and a constant `N` specifies the size of the buffer.
trait Decoder<T, const N: usize> {
    /// Decodes next elements into buffer
    ///
    /// Decodes up to `N` next elements into buffer and returns the number of decoded elements, or zero if end of
    /// stream reached. There is no guarantee about buffer element past the return value. They might be left unchanged
    /// or zeroed out by this method.
    fn decode(&mut self, buffer: &mut [T; N]) -> usize;
}

/// Stream VByte decoder
///
/// Initialized using two streams: control stream and data streams.
/// At the moment all data needs to be buffered into memory.
pub struct StreamVByteDecoder {
    control_stream: Vec<u8>,
    control_stream_pos: usize,
    data_stream: Vec<u8>,
    data_stream_pos: usize,
}

impl StreamVByteDecoder {
    pub fn new(control_stream: Vec<u8>, data_stream: Vec<u8>) -> Self {
        Self {
            control_stream,
            control_stream_pos: 0,
            data_stream,
            data_stream_pos: 0,
        }
    }
}

impl Decoder<u32, 4> for StreamVByteDecoder {
    fn decode(&mut self, buffer: &mut [u32; 4]) -> usize {
        let Some(control_word) = self.control_stream.get(self.control_stream_pos) else {
            return 0;
        };
        self.control_stream_pos += 1;
        let lengts = byte_to_4_length(*control_word);
        for (i, (item, len)) in buffer.iter_mut().zip(lengts.iter()).enumerate() {
            let len = *len as usize;
            let mut be_bytes = [0u8; 4];

            let pos = self.data_stream_pos;
            if pos + len > self.data_stream.len() {
                return i;
            }
            // Reading exactly length bytes from data stream into corresponding byte positions of a number
            be_bytes[4 - len..].copy_from_slice(&self.data_stream[pos..pos + len]);
            self.data_stream_pos += len;
            *item = u32::from_be_bytes(be_bytes);
        }
        4
    }
}

/// Decoding control byte to 4 corresponding length
///
/// The length of each integer es encoded as 2 bits: from 00 (length 1) to 11 (length 4).
fn byte_to_4_length(input: u8) -> [u8; 4] {
    [
        (input.rotate_left(2) & 0b11) + 1,
        (input.rotate_left(4) & 0b11) + 1,
        (input.rotate_left(6) & 0b11) + 1,
        (input.rotate_left(8) & 0b11) + 1,
    ]
}

// const MASK: u32 = create_mask(1, 4);

/**
Prepares shuffle mask for decoding a single `u32` using `pshufb` instruction

`len` parameter is describing the length of decoded `u32` in the input register (1-4). `offset` parameter is
describing the base offset in the register. It is the sum of all previous number lengths loaded in the input register.
*/
const fn u32_shuffle_mask(len: usize, offset: usize) -> u32 {
    const PZ: u8 = 0b10000000;
    assert!(offset < 16, "Offset should be <16");
    let offset = offset as u8;
    let p1 = 0 + offset;
    let p2 = 1 + offset;
    let p3 = 2 + offset;
    let p4 = 3 + offset;
    match len {
        1 => u32::from_be_bytes([PZ, PZ, PZ, p1]),
        2 => u32::from_be_bytes([PZ, PZ, p1, p2]),
        3 => u32::from_be_bytes([PZ, p1, p2, p3]),
        4 => u32::from_be_bytes([p1, p2, p3, p4]),
        _ => panic!("Length of u32 is 1..=4 bytes"),
    }
}

/**
Preparing shuffling masks for `pshufb` SSE instructions

`pshufb` (`_mm_shuffle_epi8()`) allows to shuffle bytes around in a `__mm128` register. Shuffle mask consist of 16
bytes. Each byte describe byte index in input register which should be copied to corresponding byte in the output
register. For addressing 16 bytes we need log(16) = 4 bits. So bits 0:3 of each byte are storing input register byte
index. MSB of each byte indicating if corresponding byte in output register should be zeroed out. 4 least significant
bits are non effective if MSB is set.

```graph
Byte offsets:              0        1        2        3        4 ... 15
Input register:         0x03     0x15     0x22     0x19     0x08 ...
                         |                 |        |        |
                         |        +--------+        |        |
                         |        |                 |        |
                         |        |          +---------------+
                         |        |          |      |
                         +-----------------------------------+
                                  |          |      |        |
Mask register:      10000000 00000010 00000100 00000011 00000000 ...
Output register:        0x00     0x22     0x08     0x19     0x03 ...
```

See [`_mm_shuffle_epi8()`][_mm_shuffle_epi8] documentation.

[_mm_shuffle_epi8]: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=shuffle_epi8&ig_expand=6097
*/
const fn shuffle_masks() -> [u128; 256] {
    let mut result = [0u128; 256];

    let mut a = 1;
    while a <= 4 {
        let mut b = 1;
        while b <= 4 {
            let mut c = 1;
            while c <= 4 {
                let mut d = 1;
                while d <= 4 {
                    let a_mask = u32_shuffle_mask(a, 0) as u128;
                    let b_mask = u32_shuffle_mask(b, a) as u128;
                    let c_mask = u32_shuffle_mask(c, a + b) as u128;
                    let d_mask = u32_shuffle_mask(d, a + b + c) as u128;
                    // counting in the index must be 0 based (eg. length of 1 is `00`, not `01`), hence `a - 1`
                    let idx = (a - 1) << 6 | (b - 1) << 4 | (c - 1) << 2 | (d - 1);
                    let mask = a_mask << 96 | b_mask << 64 | c_mask << 32 | d_mask;
                    result[idx] = mask;
                    d += 1;
                }
                c += 1;
            }
            b += 1;
        }
        a += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn check_create_mask() {
        assert_eq!(u32_shuffle_mask(1, 0), 0x8080_8000);
        assert_eq!(u32_shuffle_mask(2, 0), 0x8080_0001);

        assert_eq!(u32_shuffle_mask(1, 3), 0x8080_8003);
        assert_eq!(u32_shuffle_mask(2, 3), 0x8080_0304);
    }

    #[test]
    fn check_shuffle_masks() {
        let masks = shuffle_masks();
        assert_eq!(masks[0b00000000], 0x80808000_80808001_80808002_80808003); // Lengths 1, 1, 1, 1
        assert_eq!(masks[0b11111111], 0x00010203_04050607_08090a0b_0c0d0e0f); // Lengths 4, 1, 4, 1
        assert_eq!(masks[0b11001100], 0x00010203_80808004_05060708_80808009); // Lengths 4, 1, 4, 1
        assert_eq!(masks[0b11100100], 0x00010203_80040506_80800708_80808009); // Lengths 4, 3, 2, 1
    }

    #[test]
    fn check_encode() {
        let (control, data) = encode_values(&[0x01, 0x0100, 0x010000, 0x01000000, 0x010000]);

        assert_eq!(
            data,
            [
                0x01, //
                0x01, 0x00, //
                0x01, 0x00, 0x00, //
                0x01, 0x00, 0x00, 0x00, //
                0x01, 0x00, 0x00, //
            ]
        );

        let len = byte_to_4_length(control[0]);
        assert_eq!(len, [1, 2, 3, 4]);

        let len = byte_to_4_length(control[1]);
        assert_eq!(len, [3, 1, 1, 1]);
    }

    #[test]
    fn check_decode() {
        let input = [1, 255, 1024, 2048, 0xFF000000];
        let (control, data) = encode_values(&input);
        let mut decoder = StreamVByteDecoder::new(control, data);
        let mut buffer = [0u32; 4];

        for chunk in input.chunks(4) {
            let decoded = decoder.decode(&mut buffer);
            let len = chunk.len();
            assert_eq!(decoded, len);
            assert_eq!(buffer[..len], *chunk);
        }
    }

    fn encode_values(input: &[u32]) -> (Vec<u8>, Vec<u8>) {
        let mut encoder = StreamVByteEncoder::new(Cursor::new(vec![]), Cursor::new(vec![]));
        encoder.encode(&input).unwrap();
        let (control, data) = encoder.finish().unwrap();
        (control.into_inner(), data.into_inner())
    }
}
