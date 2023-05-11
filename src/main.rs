//! https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/
//! https://arxiv.org/abs/1209.2137
//!
use std::{
    io::{self, Write},
    mem::transmute,
};

fn main() {
    println!("Hello, world!");
}

struct StreamVByteEncoder<W> {
    data: Box<W>,
    control: Box<W>,
    control_byte: u8,
    written: u8,
}

impl<W: Write> StreamVByteEncoder<W> {
    fn new(data: W, control: W) -> Self {
        let data = Box::new(data);
        let control = Box::new(control);
        Self {
            data,
            control,
            control_byte: 0,
            written: 0,
        }
    }
    /// Compresses input data using stream algorithm
    fn compress(&mut self, input: &[u32]) -> io::Result<()> {
        for n in input {
            let bytes: [u8; 4] = unsafe { transmute(n.to_be()) };
            let length = 4 - n.leading_zeros() as u8 / 8;
            assert!(length <= 4);

            self.control_byte <<= 2;
            self.control_byte |= length - 1;

            self.data.write_all(&bytes[4 - length as usize..])?;
            self.written += 1;
            self.write_control_byte_if_needed()?;
        }
        Ok(())
    }

    fn write_control_byte_if_needed(&mut self) -> io::Result<()> {
        Ok(if self.written == 4 {
            self.control.write_all(&[self.control_byte])?;
            self.control_byte = 0;
            self.written = 0;
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        self.write_control_byte_if_needed()?;
        self.data.flush()?;
        self.control.flush()?;
        Ok(())
    }

    fn into_inner(mut self) -> io::Result<(W, W)> {
        self.flush()?;
        Ok((*self.control, *self.data))
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

        let input: &[u32] = &[0x01, 0x0100, 0x010000, 0x01000000];
        let mut encoder = StreamVByteEncoder::new(data, control);
        encoder.compress(&input).unwrap();
        let (control, data) = encoder.into_inner().unwrap();

        let data = data.into_inner();
        assert_eq!(
            data,
            [
                0x01, //
                0x01, 0x00, //
                0x01, 0x00, 0x00, //
                0x01, 0x00, 0x00, 0x00 //
            ]
        );

        let control = control.into_inner();
        let lengths = byte_to_4_length(control[0]);
        assert_eq!(lengths, [1, 2, 3, 4]);
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
