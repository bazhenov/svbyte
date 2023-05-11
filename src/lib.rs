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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

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
