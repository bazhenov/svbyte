#![feature(stdsimd)]
/*!
This library provides encoding/decoding primitives for Stream VByte encoding.

Stream VByte encoding is a SIMD accelerated algorithm of VarInt decompression. It is used in a search and database
systems as a way of efficiently store and stream large number of VarInts from a disk or main memory.

The idea behind VarInt is not to store leading bytes of a number, so large amount of relatively small numbers can be
stored in a much more compact way. VarInt encoding is frequently used with delta-encoding if numbers are stored in the
ascending order. This way all the numbers are smaller by magnitude, hence better compression.

Stream VByte working using two data streams: control stream and data stream. Control stream contains control words (1
byte each). Each control word describe length of 4 numbers in the data stream (2 bits per number, `00` - length 1,
`01` - length 2 and so on).

- [Decoding billions of integers per second through vectorization][pub] by Daniel Lemire and Leonid Boytsov.
- [Stream VByte: breaking new speed records for integer compression][blog-post] by Daniel Lemire

[pub]: https://arxiv.org/abs/1209.2137
[blog-post]: https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/
*/
use std::arch::x86_64::{_mm_loadu_epi8, _mm_shuffle_epi8, _mm_store_epi32};
use std::io::{self, BufRead, Write};

#[allow(non_camel_case_types)]
type u32x4 = [u32; 4];

/// Shuffle masks and correspinding length of encoded numbers
///
/// For more information see documentation to [`u32_shuffle_masks`]
///
/// [`u32_shuffle_masks`]: u32_shuffle_masks
const MASKS: [(u32x4, u8); 256] = u32_shuffle_masks();

const SEGMENT_MAGIC: u16 = 0x0B0D;

/// Stream VByte decoder
///
/// Initialized using two streams: control stream and data streams.
/// At the moment all data needs to be buffered into memory.
pub struct StreamVByteDecoder<R> {
    control_stream: Vec<u8>,
    control_stream_pos: usize,
    data_stream: Vec<u8>,
    data_stream_pos: usize,
    source: Box<R>,
    elements_left: usize,
}

impl<R: BufRead> StreamVByteDecoder<R> {
    pub fn new(source: R) -> Self {
        Self {
            control_stream: vec![],
            control_stream_pos: 0,
            data_stream: vec![],
            data_stream_pos: 0,
            source: Box::new(source),
            elements_left: 0,
        }
    }

    fn refill(&mut self) -> io::Result<()> {
        debug_assert!(
            self.elements_left == 0,
            "Should be 0, got: {}",
            self.elements_left
        );
        self.control_stream_pos = 0;
        self.data_stream_pos = 0;
        let result = read_segment(
            &mut self.source,
            &mut self.control_stream,
            &mut self.data_stream,
        );
        match result {
            Ok(elements) => {
                self.elements_left = elements;
                Ok(())
            }
            Err(e) => {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    self.control_stream.clear();
                    self.data_stream.clear();
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }
}

/// Reads the segment, checks segment header and copies streams into corresponding buffers
///
/// Returns the number of elements encoded in the segment
fn read_segment(input: &mut impl BufRead, cs: &mut Vec<u8>, ds: &mut Vec<u8>) -> io::Result<usize> {
    let mut buf = [0u8; 2];
    input.read_exact(&mut buf)?;
    let magic = u16::from_be_bytes(buf);

    assert!(
        magic == SEGMENT_MAGIC,
        "Expected magic: {}, got: {}",
        SEGMENT_MAGIC,
        magic,
    );

    let mut buf = [0u8; 4];
    input.read_exact(&mut buf)?;
    let number_of_elements = u32::from_be_bytes(buf) as usize;

    input.read_exact(&mut buf)?;
    let cs_length = u32::from_be_bytes(buf) as usize;

    input.read_exact(&mut buf)?;
    let ds_length = u32::from_be_bytes(buf) as usize;

    cs.resize(cs_length, 0);
    input.read_exact(&mut cs[..cs_length])?;

    ds.resize(ds_length, 0);
    input.read_exact(&mut ds[..ds_length])?;

    Ok(number_of_elements)
}

impl<R: BufRead> Decoder<u32, 4> for StreamVByteDecoder<R> {
    fn decode(&mut self, buffer: &mut u32x4) -> usize {
        if self.control_stream_pos >= self.control_stream.len() {
            self.refill().unwrap();
        }
        let Some(control_word) = self.control_stream.get(self.control_stream_pos) else {
            return 0;
        };

        let (ref mask, encoded_len) = MASKS[*control_word as usize];
        let input = &self.data_stream[self.data_stream_pos];
        unsafe {
            let mask = _mm_loadu_epi8(mask as *const u32x4 as *const i8);
            let input = _mm_loadu_epi8(input as *const u8 as *const i8);
            let answer = _mm_shuffle_epi8(input, mask);
            _mm_store_epi32(buffer as *mut u32x4 as *mut i32, answer);
        }
        let elements_decoded = self.elements_left.min(4);
        self.elements_left -= elements_decoded;
        self.data_stream_pos += encoded_len as usize;
        self.control_stream_pos += 1;
        elements_decoded
    }
}

/**
Prepares shuffle mask for decoding a single `u32` using `pshufb` instruction

`len` parameter is describing the length of decoded `u32` in the input register (1-4). `offset` parameter is
describing the base offset in the register. It is the sum of all previous number lengths loaded in the input register.
*/
const fn u32_shuffle_mask(len: usize, offset: usize) -> u32 {
    const PZ: u8 = 0b10000000;
    assert!(offset < 16, "Offset should be <16");
    let offset = offset as u8;
    let p1 = offset;
    let p2 = offset + 1;
    let p3 = offset + 2;
    let p4 = offset + 3;
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
const fn u32_shuffle_masks() -> [(u32x4, u8); 256] {
    let mut masks = [([0u32; 4], 0u8); 256];

    let mut a = 1;
    while a <= 4 {
        let mut b = 1;
        while b <= 4 {
            let mut c = 1;
            while c <= 4 {
                let mut d = 1;
                while d <= 4 {
                    // Loading in reverse order because Intel is Little Endian Machine
                    let mask = [
                        u32_shuffle_mask(a, 0),
                        u32_shuffle_mask(b, a),
                        u32_shuffle_mask(c, a + b),
                        u32_shuffle_mask(d, a + b + c),
                    ];

                    // counting in the index must be 0 based (eg. length of 1 is `00`, not `01`), hence `a - 1`
                    let idx = (a - 1) << 6 | (b - 1) << 4 | (c - 1) << 2 | (d - 1);
                    assert!(a + b + c + d <= 16);
                    masks[idx] = (mask, (a + b + c + d) as u8);
                    d += 1;
                }
                c += 1;
            }
            b += 1;
        }
        a += 1;
    }
    masks
}

/**
Stream VByte Encoder

Encodes a stream of numbers and saves them in a [`Write`] output stream.

Data format follows this structure:

```diagram
┌───────┬───────┬─────────┬─────────┬────────┬────────┐
│ MAGIC │ COUNT │ CS SIZE │ DS SIZE │ CS ... │ DS ... │
└───────┴───────┴─────────┴─────────┴────────┴────────┘
```

- `MAGIC` is always 0x0B0D;
- `COUNT` the number of elements encoded in the segment (u32);
- `CS SIZE` is the size of control stream in bytes (u32);
- `DS SIZE` is the size of data stream in bytes (u32);
- `CS` and `DS` and control and data streams.

Segment header (`MAGIC`, `CS SIZE`, `DS SIZE`) is enough to calculate the whole segment size.
Segments follows each other until EOF of a stream reached.
*/
pub struct StreamVByteEncoder<W> {
    data_stream: Vec<u8>,
    control_stream: Vec<u8>,
    output: Box<W>,
    written: usize,
}

impl<W: Write> StreamVByteEncoder<W> {
    pub fn new(output: W) -> Self {
        Self {
            data_stream: vec![],
            control_stream: vec![],
            output: Box::new(output),
            written: 0,
        }
    }
    /// Compresses input data using stream algorithm
    pub fn encode(&mut self, input: &[u32]) -> io::Result<()> {
        for n in input {
            let bytes: [u8; 4] = n.to_be_bytes();
            let length = 4 - n.leading_zeros() as u8 / 8;
            debug_assert!(1 <= length && length <= 4);

            let control_word = self.get_control_word();
            *control_word <<= 2;
            *control_word |= length - 1;
            self.written += 1;

            self.data_stream.write_all(&bytes[4 - length as usize..])?;
            self.write_segment_if_needed()?;
        }
        Ok(())
    }

    fn get_control_word(&mut self) -> &mut u8 {
        if self.written % 4 == 0 {
            self.control_stream.push(0);
        }
        self.control_stream.last_mut().unwrap()
    }

    fn write_segment_if_needed(&mut self) -> io::Result<()> {
        const MAX_SEGMENT_SIZE: usize = 16 * 1024;
        let segment_size = 2 // magic size
            + 4 // stream size
            + 4 // control stream size
            + 4 // data stream size
            + self.data_stream.len() + self.control_stream.len();
        if segment_size >= MAX_SEGMENT_SIZE {
            self.write_segment()?;

            self.written = 0;
            self.data_stream.clear();
            self.control_stream.clear();
        }
        Ok(())
    }

    fn write_segment(&mut self) -> io::Result<()> {
        let tail = self.written % 4;
        // we need to binary shift last control left if number of elements
        // not multiple of 4, otherwise last control will be misaligned
        if tail > 0 {
            let control_word = self.control_stream.last_mut().unwrap();
            *control_word <<= 2 * (4 - tail);
        }

        self.output.write_all(&SEGMENT_MAGIC.to_be_bytes())?;

        debug_assert!(self.written <= u32::MAX as usize);
        let number_of_elements = (self.written as u32).to_be_bytes();
        self.output.write_all(&number_of_elements)?;

        debug_assert!(self.control_stream.len() <= u32::MAX as usize);
        let cs_len = (self.control_stream.len() as u32).to_be_bytes();
        self.output.write_all(&cs_len)?;

        debug_assert!(self.data_stream.len() <= u32::MAX as usize);
        let ds_len = (self.data_stream.len() as u32).to_be_bytes();
        self.output.write_all(&ds_len)?;

        self.output.write_all(&self.control_stream)?;
        self.output.write_all(&self.data_stream)?;

        Ok(())
    }

    /// Returns output stream stream back to the client
    ///
    /// All pending writes are not *flushed*. It is a responsibility of a callee to flush if needed.
    pub fn finish(mut self) -> io::Result<W> {
        self.write_segment()?;
        Ok(*self.output)
    }
}

/// Represents an object that can decode a stream of data into a buffer of fixed size. A type parameter `T` specifies /// the type of the elements in the buffer, and a constant `N` specifies the size of the buffer.
trait Decoder<T: Default + Copy, const N: usize> {
    /// Decodes next elements into buffer
    ///
    /// Decodes up to `N` next elements into buffer and returns the number of decoded elements, or zero if end of
    /// stream reached. There is no guarantee about buffer element past the return value. They might be left unchanged
    /// or zeroed out by this method.
    fn decode(&mut self, buffer: &mut [T; N]) -> usize;

    fn to_vec(mut self) -> Vec<T>
    where
        Self: Sized,
    {
        let mut buffer = [Default::default(); N];
        let mut result = vec![];
        let mut len = self.decode(&mut buffer);
        while len > 0 {
            result.extend(&buffer[..len]);
            len = self.decode(&mut buffer);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::ThreadRng, thread_rng, Rng};
    use std::io::{Cursor, Seek, SeekFrom};

    #[test]
    fn check_encode() {
        let (control, data, _) = encode_values(&[0x01, 0x0100, 0x010000, 0x01000000, 0x010000]);

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
    fn check_small_functional_encode_decode() {
        let mut rng = thread_rng();
        for _ in 0..1000 {
            let len = rng.gen_range(1..20);
            check_encode_decode_cycle(&mut rng, len);
        }
    }

    #[test]
    fn check_large_functional_encode_decode() {
        let mut rng = thread_rng();
        for _ in 0..10 {
            let len = rng.gen_range(10000..20000);
            check_encode_decode_cycle(&mut rng, len);
        }
    }

    fn check_encode_decode_cycle(rng: &mut ThreadRng, len: usize) {
        let mut input: Vec<u32> = vec![];
        input.resize(len, 0);
        rng.fill(&mut input[..]);
        let (_, _, encoded) = encode_values(&input);
        let output = StreamVByteDecoder::new(encoded).to_vec();
        assert_eq!(input.len(), output.len());
        assert_eq!(output, input);
    }

    #[test]
    fn check_decode() {
        let input = [1, 255, 1024, 2048, 0xFF000000];
        let (_, _, encoded) = encode_values(&input);
        let output = StreamVByteDecoder::new(encoded).to_vec();
        assert_eq!(output.len(), output.len());
        assert_eq!(output, input);
    }

    #[test]
    fn check_create_mask() {
        assert_eq!(u32_shuffle_mask(1, 0), 0x808080_00);
        assert_eq!(u32_shuffle_mask(2, 0), 0x8080_0001);

        assert_eq!(u32_shuffle_mask(1, 3), 0x808080_03);
        assert_eq!(u32_shuffle_mask(2, 3), 0x8080_0304);
    }

    #[test]
    fn check_shuffle_masks() {
        let masks = u32_shuffle_masks();
        assert_eq!(
            // Lengths 1, 1, 1, 1
            masks[0b_00_00_00_00],
            ([0x808080_00, 0x808080_01, 0x808080_02, 0x808080_03], 4)
        );
        assert_eq!(
            // Lengths 4, 4, 4, 4
            masks[0b_11_11_11_11],
            ([0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f], 16)
        );
        assert_eq!(
            // Lengths 4, 1, 4, 1
            masks[0b_11_00_11_00],
            ([0x00010203, 0x808080_04, 0x05060708, 0x808080_09], 10)
        );
        assert_eq!(
            // Lengths 4, 3, 2, 1
            masks[0b_11_10_01_00],
            ([0x00010203, 0x80_040506, 0x8080_0708, 0x808080_09], 10)
        );
    }

    /// Creates and returns control and data stream for a given slice of numbers
    pub fn encode_values(input: &[u32]) -> (Vec<u8>, Vec<u8>, impl BufRead) {
        let mut encoder = StreamVByteEncoder::new(Cursor::new(vec![]));
        encoder.encode(&input).unwrap();
        let mut source = encoder.finish().unwrap();
        let mut cs = vec![];
        let mut ds = vec![];
        source.seek(SeekFrom::Start(0)).unwrap();
        read_segment(&mut source, &mut cs, &mut ds).unwrap();
        source.seek(SeekFrom::Start(0)).unwrap();
        (cs, ds, source)
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
}
