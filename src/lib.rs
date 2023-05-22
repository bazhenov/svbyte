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
use std::{
    arch::x86_64::{_mm_loadu_si128, _mm_shuffle_epi8, _mm_storeu_si128},
    io::{self, BufRead, Write},
    ptr::NonNull,
};

#[allow(non_camel_case_types)]
type u32x4 = [u32; 4];

/// Shuffle masks and correspinding length of encoded numbers
///
/// For more information see documentation to [`u32_shuffle_masks`]
///
/// [`u32_shuffle_masks`]: u32_shuffle_masks
const MASKS: [(u32x4, u8); 256] = u32_shuffle_masks();

/// Marker bytes of a [`SegmentHeader`]
const SEGMENT_MAGIC: u16 = 0x0B0D;

/// Lenth of [`SegmentHeader`] in bytes
const SEGMENT_HEADER_LENGTH: usize = 14;

/// Provides facility for reading segments
///
/// Each segment contains elements (integers) in encoded format. Each [`next`] method call
/// moves this objects to the next segment.
///
/// ## Motivation
/// This trait exists to abstract [`DecodeCursor`] from logic of reading segments. If all the segments are
/// in memory the most efficient way of decoding is decoding `[u8]` slices in memory. This maximize the
/// decoding speed because no memory copy should be done. In case segments data are on the file systems,
/// there should exists logic for reading next segment in a memory buffer. In latter case it's more
/// appropriate to read segments one by one in a memory buffer of predefined size. [`Segments`] trait
/// and its 2 base implementations: [`MemorySegments`] and [`BufReadSegments`] are providing those facilities.
pub trait Segments {
    /// Moves to the next segment and return number of the elements encoded in the segment as well as
    /// control and data streams
    fn next(&mut self) -> io::Result<Option<(usize, &[u8], &[u8])>>;
}

/// Reads segment from underlying [`BufRead`]
pub struct BufReadSegments<R> {
    source: R,
    control_stream: Vec<u8>,
    data_stream: Vec<u8>,
}

impl<R> BufReadSegments<R> {
    pub fn new(source: R) -> Self {
        Self {
            source,
            control_stream: vec![],
            data_stream: vec![],
        }
    }
}

impl<R: BufRead> Segments for BufReadSegments<R> {
    fn next(&mut self) -> io::Result<Option<(usize, &[u8], &[u8])>> {
        let result = read_segment(
            &mut self.source,
            &mut self.control_stream,
            &mut self.data_stream,
        );
        match result {
            Ok(elements) => Ok(Some((elements, &self.control_stream, &self.data_stream))),
            Err(e) => {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    Ok(None)
                } else {
                    Err(e)
                }
            }
        }
    }
}

/// [`Segments`] implementation with all segment data in memory
pub struct MemorySegments<'a> {
    data: &'a [u8],
}

impl<'a> MemorySegments<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }
}

impl<'a> Segments for MemorySegments<'a> {
    fn next(&mut self) -> io::Result<Option<(usize, &[u8], &[u8])>> {
        if self.data.is_empty() {
            return Ok(None);
        }

        let segment = SegmentHeader::parse(self.data);
        let control_stream =
            &self.data[SEGMENT_HEADER_LENGTH..SEGMENT_HEADER_LENGTH + segment.cs_length];
        let data_stream = &self.data[SEGMENT_HEADER_LENGTH + segment.cs_length
            ..SEGMENT_HEADER_LENGTH + segment.cs_length + segment.ds_length];
        self.data = &self.data[SEGMENT_HEADER_LENGTH + segment.cs_length + segment.ds_length..];

        Ok(Some((segment.count, control_stream, data_stream)))
    }
}

/// Stream VByte decoding cursor
///
/// Cursor allows to decode stream of elements using [`Segments`] as a source
/// of decoding data. Implements [`Decoder`] trait.
pub struct DecodeCursor<S: Segments> {
    elements_left: usize,
    control_stream: *const u8,
    data_stream: *const u8,
    segments: S,
}

impl<S: Segments> DecodeCursor<S> {
    pub fn new(segments: S) -> io::Result<Self> {
        Ok(Self {
            elements_left: 0,
            control_stream: NonNull::dangling().as_ptr(),
            data_stream: NonNull::dangling().as_ptr(),
            segments,
        })
    }

    #[inline(never)]
    fn refill(&mut self) -> io::Result<usize> {
        debug_assert!(
            self.elements_left == 0,
            "Should be 0, got: {}",
            self.elements_left
        );

        if let Some((elements, cs, ds)) = self.segments.next()? {
            assert!(
                cs.len() * 4 >= elements,
                "Invalid control stream length. Expected: {}, got: {}",
                (elements + 3) / 4,
                cs.len()
            );
            assert!(
                ds.len() >= elements,
                "Invalid data stream length. Expected: >={}, got: {}",
                elements,
                ds.len()
            );
            self.control_stream = cs.as_ptr();
            self.data_stream = ds.as_ptr();
            self.elements_left = elements;
            Ok(elements)
        } else {
            Ok(0)
        }
    }
}

/// Segment Header
///
/// Each segment starts with a header described in the [`EncodeCursor`] documentation.
#[derive(Debug, PartialEq)]
struct SegmentHeader {
    count: usize,
    cs_length: usize,
    ds_length: usize,
}

impl SegmentHeader {
    fn new(count: usize, cs_size: usize, ds_size: usize) -> Self {
        Self {
            count,
            cs_length: cs_size,
            ds_length: ds_size,
        }
    }

    fn parse(input: &[u8]) -> Self {
        assert!(
            input.len() >= SEGMENT_HEADER_LENGTH,
            "Expected slice of len >={}, got: {}",
            SEGMENT_HEADER_LENGTH,
            input.len()
        );
        let input = &input[..SEGMENT_HEADER_LENGTH];

        let magic = u16::from_be_bytes(input[0..2].try_into().unwrap());
        let count = u32::from_be_bytes(input[2..6].try_into().unwrap()) as usize;
        let cs_length = u32::from_be_bytes(input[6..10].try_into().unwrap()) as usize;
        let ds_length = u32::from_be_bytes(input[10..14].try_into().unwrap()) as usize;

        assert!(
            magic == SEGMENT_MAGIC,
            "Expected magic: {}, got: {}",
            SEGMENT_MAGIC,
            magic,
        );

        Self {
            count,
            cs_length,
            ds_length,
        }
    }

    fn write(&self, out: &mut dyn Write) -> io::Result<()> {
        out.write_all(&SEGMENT_MAGIC.to_be_bytes())?;

        debug_assert!(self.count <= u32::MAX as usize);
        let number_of_elements = (self.count as u32).to_be_bytes();
        out.write_all(&number_of_elements)?;

        debug_assert!(self.cs_length <= u32::MAX as usize);
        let cs_len = (self.cs_length as u32).to_be_bytes();
        out.write_all(&cs_len)?;

        debug_assert!(self.ds_length <= u32::MAX as usize);
        let ds_len = (self.ds_length as u32).to_be_bytes();
        out.write_all(&ds_len)?;

        Ok(())
    }
}

/// Reads the segment, checks segment header and copies streams into corresponding buffers
///
/// Returns the number of elements encoded in the segment
fn read_segment(input: &mut impl BufRead, cs: &mut Vec<u8>, ds: &mut Vec<u8>) -> io::Result<usize> {
    let mut buf = [0u8; SEGMENT_HEADER_LENGTH];
    input.read_exact(&mut buf)?;
    let header = SegmentHeader::parse(&buf);

    cs.resize(header.cs_length, 0);
    input.read_exact(&mut cs[..header.cs_length])?;

    ds.resize(header.ds_length, 0);
    input.read_exact(&mut ds[..header.ds_length])?;

    Ok(header.count)
}

impl<S: Segments> Decoder<u32> for DecodeCursor<S> {
    fn decode(&mut self, buffer: &mut [u32]) -> io::Result<usize> {
        /// Number of decoded elements per iteration
        const DECODE_WIDTH: usize = 4;
        assert!(
            buffer.len() >= DECODE_WIDTH,
            "Buffer should be at least {} elements long",
            DECODE_WIDTH
        );
        if self.elements_left == 0 && self.refill()? == 0 {
            return Ok(0);
        }

        let mut iterations = buffer.len() / DECODE_WIDTH;
        iterations = iterations.min((self.elements_left + DECODE_WIDTH - 1) / DECODE_WIDTH);
        let decoded = iterations * DECODE_WIDTH;

        let mut data_stream = self.data_stream;
        let mut control_stream = self.control_stream;
        let mut buffer = buffer.as_mut_ptr() as *mut u32x4;

        // Decode loop unrolling
        const UNROLL_FACTOR: usize = 8;
        while iterations >= UNROLL_FACTOR {
            for _ in 0..UNROLL_FACTOR {
                let encoded_len = simd_decode(data_stream, control_stream, buffer);
                data_stream = data_stream.wrapping_add(encoded_len as usize);
                buffer = buffer.wrapping_add(1);
                control_stream = control_stream.wrapping_add(1);
            }

            iterations -= UNROLL_FACTOR;
        }

        // Tail decode
        for _ in 0..iterations {
            let encoded_len = simd_decode(data_stream, control_stream, buffer);
            data_stream = data_stream.wrapping_add(encoded_len as usize);
            buffer = buffer.wrapping_add(1);
            control_stream = control_stream.wrapping_add(1);
        }

        self.control_stream = control_stream;
        self.data_stream = data_stream;
        let decoded = decoded.min(self.elements_left);
        self.elements_left -= decoded;
        Ok(decoded)
    }
}

/// Decoding SIMD kernel using SSE intrinsics
///
/// Types of this function tries to implement safety guardrails as much as possible. Namely:
/// `output` - is a reference to the buffer of 4 u32 values;
/// `input` - is a reference to u8 array of unspecified length (`control_word` speciefies how much will be decoded);
//
/// Technically the encoded length can be calculated from control word directly using horizontal 2-bit sum
/// ```rust,ignore
/// let result = *control_word;
/// let result = ((result & 0b11001100) >> 2) + (result & 0b00110011);
/// let result = (result >> 4) + (result & 0b1111) + 4;
/// ```
/// Unfortunatley, this approach is slower then memoized length. There is a mention of this approach can be faster
/// when using `u32` control words, which implies decoding a batch of size 16[^1].
///
/// [^1]: [Bit hacking versus memoization: a Stream VByte example](https://lemire.me/blog/2017/11/28/bit-hacking-versus-memoization-a-stream-vbyte-example/)
#[inline]
fn simd_decode(input: *const u8, control_word: *const u8, output: *mut u32x4) -> u8 {
    unsafe {
        let (ref mask, encoded_len) = MASKS[*control_word as usize];
        let mask = _mm_loadu_si128(mask.as_ptr().cast());
        let input = _mm_loadu_si128(input.cast());
        let answer = _mm_shuffle_epi8(input, mask);
        _mm_storeu_si128(output.cast(), answer);

        encoded_len
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

`pshufb` SSE instruction visualization.

```graph
  Byte offsets:           0        1        2        3        4
                  ┌────────┬────────┬────────┬────────┬────────┬───┐
Input Register:   │   0x03 │   0x15 │   0x22 │   0x19 │   0x08 │...│
                  └────▲───┴────────┴────▲───┴────▲───┴────▲───┴───┘
                       │        ┌────────┘        │        │
                       │        │        ┌─────────────────┘
                       │        │        │        │
                       └───────────────────────────────────┐
                                │        │        │        │
                  ┌────────┬────┴───┬────┴───┬────┴───┬────┴───┬───┐
  Mask Register:  │   0x80 │   0x02 │   0x04 │   0x03 │   0x00 │...│
                  ├────────┼────────┼────────┼────────┼────────┼───┤
Output Register:  │   0x00 │   0x22 │   0x08 │   0x19 │   0x03 │...│
                  └────────┴────────┴────────┴────────┴────────┴───┘
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
- `COUNT` the number of elements encoded in the segment (`u32`);
- `CS SIZE` is the size of control stream in bytes (`u32`);
- `DS SIZE` is the size of data stream in bytes (`u32`);
- `CS` and `DS` and control and data streams.

Segment header (`MAGIC`, `COUNT`, `CS SIZE`, `DS SIZE`) is enough to calculate the whole segment size.
Segments follows each other until EOF of a stream reached.
*/
pub struct EncodeCursor<W> {
    data_stream: Vec<u8>,
    control_stream: Vec<u8>,
    output: Box<W>,
    written: usize,
}

impl<W: Write> EncodeCursor<W> {
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
            let length = length.max(1);
            debug_assert!((1..=4).contains(&length));

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
        const MAX_SEGMENT_SIZE: usize = 8 * 1024;
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
            for _ in 0..(4 - tail) {
                self.data_stream.write_all(&[0])?;
            }
        }

        let header = SegmentHeader::new(
            self.written,
            self.control_stream.len(),
            self.data_stream.len(),
        );
        header.write(&mut self.output)?;

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

/// Represents an object that can decode a stream of data into a buffer of fixed size. A type parameter `T` specifies /// the type of the elements in the buffer.
pub trait Decoder<T: Copy + From<u8>> {
    /// Decodes next elements into buffer
    ///
    /// Decodes next elements into buffer and returns the number of decoded elements, or zero if and end of the
    /// stream is reached. There is no guarantee about buffer element past the return value. They might be
    /// left unchanged or zeroed out by this method.
    fn decode(&mut self, buffer: &mut [T]) -> io::Result<usize>;

    fn to_vec(mut self) -> io::Result<Vec<T>>
    where
        Self: Sized,
    {
        let mut buffer = [0u8.into(); 128];
        let mut result = vec![];
        let mut len = self.decode(&mut buffer)?;
        while len > 0 {
            result.extend(&buffer[..len]);
            len = self.decode(&mut buffer)?;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::ThreadRng, thread_rng, Rng, RngCore};
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
                0x00, 0x00, 0x00, // 3 bytes a padding so segment size is multiple of 4
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
        let input: Vec<u32> = generate_random_data(rng, len);
        let (_, _, encoded) = encode_values(&input);
        let output = DecodeCursor::new(MemorySegments::new(&encoded.into_inner()))
            .unwrap()
            .to_vec()
            .unwrap();
        assert_eq!(input.len(), output.len());
        let chunk_size = 4;
        for (i, (input, output)) in input
            .chunks(chunk_size)
            .zip(output.chunks(chunk_size))
            .enumerate()
        {
            assert_eq!(input, output, "Arrays differs position {}", i * chunk_size);
        }
    }

    #[test]
    fn check_decode() {
        let input = [1, 255, 1024, 2048, 0xFF000000];
        let (_, _, encoded) = encode_values(&input);
        let output = DecodeCursor::new(MemorySegments::new(&encoded.into_inner()))
            .unwrap()
            .to_vec()
            .unwrap();
        assert_eq!(output.len(), output.len());
        assert_eq!(output, input);
    }

    #[allow(clippy::unusual_byte_groupings)]
    #[test]
    fn check_create_mask() {
        assert_eq!(u32_shuffle_mask(1, 0), 0x808080_00);
        assert_eq!(u32_shuffle_mask(2, 0), 0x8080_0001);

        assert_eq!(u32_shuffle_mask(1, 3), 0x808080_03);
        assert_eq!(u32_shuffle_mask(2, 3), 0x8080_0304);
    }

    #[allow(clippy::unusual_byte_groupings)]
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

    #[test]
    fn check_header_format() {
        let expected = SegmentHeader::new(3, 1, 2);
        let mut out = vec![];

        expected.write(&mut out).unwrap();
        let header = SegmentHeader::parse(&out);
        assert_eq!(header, expected);
    }

    /// Creates and returns control and data stream for a given slice of numbers
    pub fn encode_values(input: &[u32]) -> (Vec<u8>, Vec<u8>, Cursor<Vec<u8>>) {
        let mut encoder = EncodeCursor::new(Cursor::new(vec![]));
        encoder.encode(input).unwrap();
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

    /// Generates "weighed" dataset fortesting purposes
    ///
    /// "Weighted" basically means that there is equal number of elements (in probabilistic terms)
    /// with different length in varint encoding.
    fn generate_random_data(rng: &mut ThreadRng, size: usize) -> Vec<u32> {
        let mut input = vec![];
        input.resize_with(size, || match rng.gen_range(1..=4) {
            1 => rng.next_u32() % (0xFF + 1),
            2 => rng.next_u32() % (0xFFFF + 1),
            3 => rng.next_u32() % (0xFFFFFF + 1),
            _ => rng.next_u32(),
        });
        input
    }
}
