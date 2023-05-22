use std::{
    fs::File,
    io::{self, BufReader, BufWriter},
    time::Instant,
};
use svbyte::{BufReadSegments, DecodeCursor, Decoder, EncodeCursor};

fn main() -> io::Result<()> {
    let file_name = "./result.bin";
    let mut encoder = EncodeCursor::new(BufWriter::new(File::create(file_name)?));

    let start = Instant::now();
    let elements = 1_000_000;
    for i in 1..=elements {
        encoder.encode(&[i])?;
    }
    println!(
        "Encoded {} elements in {}ms",
        elements,
        start.elapsed().as_millis()
    );

    encoder.finish()?;

    let segments = BufReadSegments::new(BufReader::new(File::open(file_name)?));
    let mut decoder = DecodeCursor::new(segments)?;

    let start = Instant::now();
    let mut buffer = [0u32; 128];
    let mut decoded = decoder.decode(&mut buffer)?;
    let mut sum = 0u64;
    while decoded > 0 {
        sum += buffer[..decoded].iter().sum::<u32>() as u64;
        decoded = decoder.decode(&mut buffer)?;
    }

    // The sum of first N elements (arithmetic progression) should be N * (N + 1) / 2
    let elements = elements as u64;
    assert!(sum == elements * (elements + 1) / 2);

    println!(
        "Decoded {} elements in {}ms",
        elements,
        start.elapsed().as_millis()
    );

    Ok(())
}
