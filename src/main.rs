//! https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/
//! https://arxiv.org/abs/1209.2137
//!
use std::io::{self, Write};

fn main() {
    println!("Hello, world!");
}

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

    pub fn flush(&mut self) -> io::Result<()> {
        self.ensure_control_word_written()?;
        self.data_stream.flush()?;
        self.control_stream.flush()?;
        Ok(())
    }

    /// Returns control and data stream back to the client
    ///
    /// Flushes all pending writes and returns tuple of two streams `(control, data)`.
    pub fn into_inner(mut self) -> io::Result<(W, W)> {
        // We need to pad last control word with zero bits if number of elements
        // not multiple of 4, otherwise last control word will not be written
        {
            self.control_word <<= 2 * (4 - self.written);
            self.written = 4;
        }

        self.flush()?;
        Ok((*self.control_stream, *self.data_stream))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn check_compress() {
        let data = Cursor::new(vec![]);
        let control = Cursor::new(vec![]);

        let input: &[u32] = &[0x01, 0x0100, 0x010000, 0x01000000, 0x010000];
        let mut encoder = StreamVByteEncoder::new(data, control);
        encoder.encode(&input).unwrap();
        let (control, data) = encoder.into_inner().unwrap();

        let data = data.into_inner();
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

        let control = control.into_inner();
        let lengths = byte_to_4_length(control[0]);
        assert_eq!(lengths, [1, 2, 3, 4]);

        let lengths = byte_to_4_length(control[1]);
        assert_eq!(lengths, [3, 1, 1, 1]);
    }

    /// Decoding control byte to 4 corresponding length
    ///
    /// The length of each integer es encoded as 2 bits: from 00 (length 1) to 11 (length 4).
    fn byte_to_4_length(input: u8) -> [u8; 4] {
        [
            (input >> 6 & 0b11) + 1,
            (input >> 4 & 0b11) + 1,
            (input >> 2 & 0b11) + 1,
            (input >> 0 & 0b11) + 1,
        ]
    }
}
